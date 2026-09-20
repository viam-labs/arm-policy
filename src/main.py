import asyncio
from viam.module.module import Module
from models.policy import Policy as PolicyModel


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
