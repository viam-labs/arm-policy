import asyncio

from viam.module.module import Module

import viam_arm_policy # noqa: F401 - importing registers the policy model

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
