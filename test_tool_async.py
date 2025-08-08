#!/usr/bin/env python3
import asyncio
from main import content_analysis_tool

async def test():
    try:
        result = await content_analysis_tool.ainvoke({'content': 'test'})
        print('SUCCESS:', list(result.keys()))
        return result
    except Exception as e:
        print('ERROR:', e)
        return None

if __name__ == "__main__":
    asyncio.run(test())