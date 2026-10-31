from getpass import getpass
from app.database import Base,SessionLocal,engine
from app.models import User
from app.services.auth import hash_password

Base.metadata.create_all(bind=engine)

db=SessionLocal()

try:
    username=input("Username: ").strip()
    email=input("Email: ").strip() or None
    password=getpass("Password: ")

    if db.query(User).filter(User.username==username).first():
        raise SystemExit("Username sudah ada.")

    db.add(
        User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role="admin"
        )
    )

    db.commit()
    print("User berhasil dibuat.")
finally:
    db.close()