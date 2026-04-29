import asyncio
import logging
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketingAgent")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

class MarketingContent(BaseModel):
    topic: str = Field(..., description="选定的营销热点主题")
    copywriting: str = Field(..., description="小红书/抖音风格的爆款文案")
    prompt_for_image: str = Field(..., description="用于生成配图的英文 Prompt")
    image_url: Optional[str] = None
    status: str = "draft"

class UserComment(BaseModel):
    user_id: str
    content: str
    intent: Optional[str] = None 

class StrategyAgent:
    async def generate_content(self, niche: str) -> MarketingContent:
        logger.info(f"[StrategyAgent] 正在利用长链推理生成 '{niche}' 领域的营销方案...")
        system_prompt = """你是一个顶级的私域流量营销操盘手。生成极具爆款潜质的小红书文案与配图英文Prompt。严格按JSON输出：{"topic": "话题", "copywriting": "文案", "prompt_for_image": "生图提示词"}"""
        response = await client.chat.completions.create(
            model="gpt-4o", response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"目标领域：{niche}。"}],
            temperature=0.8
        )
        content = MarketingContent.model_validate_json(response.choices[0].message.content)
        logger.info(f"[StrategyAgent] 锁定话题 '{content.topic}'")
        return content

class MultimodalAgent:
    async def generate_assets(self, content: MarketingContent) -> MarketingContent:
        logger.info("[MultimodalAgent] 正在调用云端生图接口构建视觉资产...")
        try:
            res = await client.images.generate(model="dall-e-3", prompt=content.prompt_for_image, size="1024x1024", n=1)
            content.image_url = res.data[0].url
            content.status = "ready_to_publish"
        except Exception as e:
            logger.error(f"生图失败: {e}")
            content.status = "text_only"
        return content

class InteractionAgent:
    async def handle_comment(self, comment: UserComment):
        prompt = f'分析评论意图(purchase/consult/negative/chat): "{comment.content}"'
        res = await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], max_tokens=10)
        comment.intent = res.choices[0].message.content.strip().lower()
        
        actions = {
            "purchase": "[Action - 私域转化] 发送购买链接与企微名片",
            "consult": "[Action - 私域引流] 发送干货诱导加微",
            "negative": "[Action - 危机公关] 记录工单待人工介入"
        }
        logger.info(actions.get(comment.intent, "[Action - 增加权重] 自动点赞并互动") + f" <- '{comment.content}'")

async def main():
    print("\n" + "="*50 + "\n🚀 Agent 流水线启动 (接入大模型模式)\n" + "="*50)
    strategy, multimodal, interaction = StrategyAgent(), MultimodalAgent(), InteractionAgent()

    content = await strategy.generate_content("秋季露营装备")
    final_content = await multimodal.generate_assets(content)
    
    print("\n" + "-"*50)
    logger.info(f"文案预览:\n{final_content.copywriting}\n配图链接: {final_content.image_url}")
    print("-" * 50)
    
    mock_comments = [
        UserComment(user_id="U1", content="这套帐篷链接有吗？多少钱？"),
        UserComment(user_id="U2", content="新手第一次露营，防潮垫买什么样的比较好？")
    ]
    await asyncio.gather(*(interaction.handle_comment(c) for c in mock_comments))

if __name__ == "__main__":
    asyncio.run(main())
