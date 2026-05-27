"""Dramatiq broker configuration."""

import os

import dramatiq
from dramatiq.brokers.redis import RedisBroker

broker = RedisBroker(url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
dramatiq.set_broker(broker)
