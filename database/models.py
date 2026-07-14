from sqlalchemy import Column, Integer, String, Float, DateTime
from database.connection import Base
class TrainingRun(Base):

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)

    model_name = Column(String)

    epochs = Column(Integer)

    train_loss = Column(Float)

    val_loss = Column(Float)

    ssim = Column(Float)

    psnr = Column(Float)

    rmse = Column(Float)

    mae = Column(Float)

    checkpoint = Column(String)

    created_at = Column(DateTime)