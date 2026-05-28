import json
import base64
import os
import httpx
import urllib.parse
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from dotenv import load_dotenv
import jwt

load_dotenv()

# ------------------ Конфигурация ------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./void_chat.db")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="VOID AI Chat", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()

# ------------------ Модели БД ------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    avatar = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    chats = relationship("Chat", back_populates="owner")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True)
    title = Column(String, default="Новый чат")
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", order_by="Message.timestamp",
                            cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"))
    is_user = Column(Boolean)
    text = Column(Text)
    has_attachment = Column(Boolean, default=False)
    attachment_data = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

Base.metadata.create_all(bind=engine)

# ------------------ Зависимости ------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
                     db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Недействительный токен")
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="Пользователь не найден")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Срок действия токена истёк")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Недействительный токен")

# ------------------ Обработчики ошибок ------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка сервера: {str(exc)}"})

# ------------------ Pydantic схемы ------------------
class UserCreate(BaseModel):
    username: str
    password: str

class MessageCreate(BaseModel):
    chat_id: str
    is_user: bool
    text: str
    has_attachment: bool = False
    attachment_data: Optional[str] = None

class UserOut(BaseModel):
    id: int
    username: str
    avatar: str

# ------------------ AI-логика (умеет принимать изображения) ------------------
async def ask_ai(prompt: str, history: list = None, attachment_content: str = None,
                 attachments: list = None) -> str:
    """ attachments: список словарей [{type, data (base64 url), mime, name}] """
    if not OPENROUTER_API_KEY:
        return "🤖 API ключ не настроен."

    # Если это генерация изображения, не вызываем текстовую модель
    if any(word in prompt.lower() for word in ['нарисуй', 'сгенерируй', 'изображение', 'картинк', 'рисунок']):
        return "[IMAGE_GENERATED]"

    system_msg = ("Ты — VOID, персональный AI-ассистент компании SyreksAI. "
                  "Отвечай на русском языке, кратко и полезно.")

    messages = [{"role": "system", "content": system_msg}]
    if history:
        for h in history[-10:]:
            messages.append({"role": "user" if h["is_user"] else "assistant", "content": h["text"]})

    # Формируем содержимое последнего сообщения пользователя
    user_content = [{"type": "text", "text": prompt}]

    # Обрабатываем вложения
    if attachments:
        for att in attachments:
            if att.get("type") == "image" and att.get("data"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": att["data"]}   # data уже содержит data:image/...;base64,...
                })
            elif att.get("type") == "text" and att.get("content"):
                user_content.append({
                    "type": "text",
                    "text": f"\n\n📄 Файл «{att.get('name','')}»:\n{att['content']}"
                })
    elif attachment_content:
        # старый формат (строка) – просто добавляем текст
        user_content[0]["text"] += f"\n\nСодержимое файла:\n{attachment_content}"

    messages.append({"role": "user", "content": user_content})

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "google/gemini-2.0-flash-001", "messages": messages}
            )
            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return "🤖 Ошибка AI"
        except Exception:
            return "🤖 Ошибка соединения."

# ------------------ Анализ файлов ------------------
async def analyze_image_with_ai(image_bytes: bytes, filename: str) -> str:
    if not OPENROUTER_API_KEY:
        return "[API ключ не настроен]"
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    mime = "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "Опиши подробно, что на изображении. Если есть текст — извлеки его. Отвечай на русском."},
                            {"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
                        ]
                    }]
                }
            )
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return f"[Не удалось проанализировать: {filename}]"
        except Exception:
            return f"[Ошибка анализа: {filename}]"

async def analyze_text_file(content: bytes, filename: str) -> str:
    try:
        text = content.decode('utf-8', errors='ignore')
        if len(text) > 10000:
            text = text[:10000] + "\n\n[Текст обрезан]"
        return f"Содержимое {filename}:\n\n{text}"
    except Exception:
        return f"[Не удалось прочитать: {filename}]"

# ------------------ Авторизация ------------------
@app.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Имя пользователя и пароль обязательны")
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Пользователь уже существует")
    hashed = pwd_context.hash(user.password)
    avatar = f"https://api.dicebear.com/7.x/initials/svg?seed={user.username}"
    new_user = User(username=user.username, hashed_password=hashed, avatar=avatar)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return new_user

@app.post("/token")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Неверное имя пользователя или пароль")
    payload = {"sub": user.username,
               "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token,
            "user": {"id": user.id, "username": user.username, "avatar": user.avatar}}

@app.get("/api/me", response_model=UserOut)
def get_me(user: User = Depends(get_current_user)):
    return user

# ------------------ Чаты ------------------
@app.get("/api/chats")
def get_chats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.created_at.desc()).all()
    return [{"id": c.id, "title": c.title, "messages_count": len(c.messages)} for c in chats]

@app.post("/api/chats")
def create_chat(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat_id = str(int(datetime.utcnow().timestamp() * 1000))
    chat = Chat(id=chat_id, title="Новый чат", user_id=user.id)
    db.add(chat); db.commit()
    return {"id": chat_id, "title": "Новый чат"}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    db.delete(chat); db.commit()
    return {"status": "ok"}

# ------------------ Сообщения ------------------
@app.get("/api/chats/{chat_id}/messages")
def get_messages(chat_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    msgs = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()
    return [{"id": m.id, "is_user": m.is_user, "text": m.text,
             "has_attachment": m.has_attachment, "attachment_data": m.attachment_data,
             "timestamp": m.timestamp.isoformat()} for m in msgs]

@app.post("/api/messages")
async def send_message(msg: MessageCreate, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == msg.chat_id, Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")

    # Сохраняем сообщение пользователя
    user_msg = Message(
        chat_id=msg.chat_id, is_user=msg.is_user, text=msg.text,
        has_attachment=msg.has_attachment, attachment_data=msg.attachment_data
    )
    db.add(user_msg)

    if len(chat.messages) <= 1 and msg.is_user:
        clean_text = msg.text.strip()
        if clean_text:
            chat.title = clean_text[:30] + ("..." if len(clean_text) > 30 else "")
    db.commit()

    if msg.is_user:
        # Проверка на сохранённое сгенерированное изображение
        if msg.has_attachment and msg.attachment_data:
            try:
                data = json.loads(msg.attachment_data)
                if data.get('type') == 'image' and data.get('url'):
                    return {"status": "ok", "ai_response": "🖼️"}
            except Exception:
                pass

        # Получаем историю
        history = db.query(Message).filter(Message.chat_id == msg.chat_id)\
                    .order_by(Message.timestamp).all()
        history_dicts = [{"is_user": m.is_user, "text": m.text} for m in history[-10:]]

        # Парсим attachment_data – может быть новый формат (массив объектов)
        attachments_list = None
        if msg.attachment_data:
            try:
                parsed = json.loads(msg.attachment_data)
                if isinstance(parsed, list):
                    attachments_list = parsed
            except Exception:
                pass

        ai_text = await ask_ai(
            prompt=msg.text,
            history=history_dicts,
            attachment_content=msg.attachment_data if not attachments_list else None,
            attachments=attachments_list
        )

        ai_msg = Message(chat_id=msg.chat_id, is_user=False, text=ai_text)
        db.add(ai_msg); db.commit()
        return {"status": "ok", "ai_response": ai_text}

    return {"status": "ok"}

# ------------------ Загрузка файлов ------------------
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    content = await file.read()
    filename = file.filename

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
        analysis = await analyze_image_with_ai(content, filename)
        return {"filename": filename, "analysis": analysis, "type": "image"}
    else:
        analysis = await analyze_text_file(content, filename)
        return {"filename": filename, "analysis": analysis, "type": "file"}

# ------------------ Генерация изображений ------------------
@app.post("/api/generate-image")
async def generate_image(prompt: str):
    encoded_prompt = urllib.parse.quote(prompt)
    gen_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            resp = await client.get(gen_url)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                file_id = os.path.basename(tmp_path)
                return {
                    "status": "ok",
                    "image_url": f"/api/image/{file_id}",
                    "backup_url": f"https://pollinations.ai/p/{encoded_prompt}",
                    "prompt": prompt
                }
        except Exception as e:
            print(f"Download error: {e}")

    return {
        "status": "ok",
        "image_url": gen_url,
        "backup_url": f"https://pollinations.ai/p/{encoded_prompt}",
        "prompt": prompt
    }

@app.get("/api/image/{file_id}")
async def serve_image(file_id: str):
    file_path = os.path.join(tempfile.gettempdir(), file_id)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Изображение не найдено")

# ------------------ Отдача фронтенда ------------------
@app.get("/", response_class=HTMLResponse)
async def serve():
    with open("ai.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())