# 雷石家用K歌智能助手

基于 LangChain + RAG + Agent 的智能问答系统，支持设备帮助、歌曲推荐、销售查询、售后工单。



## 环境依赖

### 1. Python 版本
- Python 3.10 或更高版本

### 2. 安装依赖
pip install -r requirements.txt

第一步：在env中配置 API Key
DEEPSEEK_API_KEY=your-deepseek-api-key
DASHSCOPE_API_KEY=your-dashscope-api-key



第二步：构建索引
bash
python rag.py
执行后会生成三个向量库和 BM25 索引，输出：


第三步：启动后端服务
bash
python agent.py


第四步：打开前端
用浏览器打开 ls.html 文件即可开始对话。




