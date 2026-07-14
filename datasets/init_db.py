from database.connection import Base, engine

import database.models

Base.metadata.create_all(bind=engine)

print("Database created.")