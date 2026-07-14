# 导入
from pathlib import Path
import re
import jieba
import numpy as np
import logging
import warnings
import pickle
import hashlib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# 屏蔽冗余日志
logging.getLogger("jieba").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class RagSystem:
    def __init__(self, api_key: str, docs_dir: str = 'docs'):
        self.api_key = api_key
        self.docs_dir = Path(docs_dir)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        # 原始文档、分块结果
        self.documents = []
        self.doc_chunks = []
        # 制造业停用词
        self.stop_words = [
            "的", "和", "与", "及", "应", "必须", "按照", "进行", "执行",
            "所有", "本", "一", "二", "三", "四", "五", "六", "七", "八",
            "九", "十", "个", "处", "时", "日", "月", "年", "每", "可",
            "无", "有", "于", "至", "等", "次", "点", "位", "部", "箱",
            "米", "元", "先", "后", "上", "下", "中", "内", "外", "以上",
            "以下", "此处", "同时", "可以", "能够", "需要", "应当", "相关",
            "对应", "存在", "全部", "整体", "各类", "人员", "规定", "文档"
        ]

        # TF-IDF向量对象
        self.tfidf_model = None
        self.tfidf_matrix = None

        # 缓存全局配置
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_version = "v1.1"

        # 缓存文件路径
        self.chunk_cache_path = self.cache_dir / "doc_chunk.pkl"
        self.tfidf_model_path = self.cache_dir / "tfidf_model.pkl"
        self.tfidf_matrix_path = self.cache_dir / "tfidf_matrix.pkl"
        self.meta_cache_path = self.cache_dir / "cache_meta.pkl"

        # TF-IDF超参快照（缓存校验用）
        self.tfidf_config = {
            "max_features": 2000,
            "min_df": 2,
            "max_df": 0.9,
            "ngram_range": (1, 2),
            "sublinear_tf": True
        }

    def get_docs_md5_hash(self) -> str:
        """计算文档目录MD5，判断文件是否修改"""
        file_info_list = []
        for file in self.docs_dir.glob("*.txt"):
            try:
                stat = os.stat(file)
                file_info_list.append(f"{str(file)}|{stat.st_mtime}")
            except Exception:
                continue
        concat_str = "||".join(file_info_list) + self.cache_version + str(self.tfidf_config)
        return hashlib.md5(concat_str.encode("utf-8")).hexdigest()

    def clear_all_cache(self):
        """清空全部缓存文件"""
        cache_files = [
            self.chunk_cache_path,
            self.tfidf_model_path,
            self.tfidf_matrix_path,
            self.meta_cache_path
        ]
        for f in cache_files:
            if f.exists():
                f.unlink()
        print("✅ 全部缓存文件已清理完成")

    def load_meta(self) -> dict | None:
        """加载缓存元数据"""
        if not self.meta_cache_path.exists():
            return None
        try:
            with open(self.meta_cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"元数据读取异常: {e}")
            return None

    def save_meta(self, doc_hash: str):
        """写入元数据快照"""
        meta = {
            "version": self.cache_version,
            "doc_hash": doc_hash,
            "tfidf_config": self.tfidf_config
        }
        try:
            with open(self.meta_cache_path, "wb") as f:
                pickle.dump(meta, f)
        except Exception as e:
            print(f"元数据写入失败，不影响主流程：{e}")

    def is_cache_valid(self) -> bool:
        """校验缓存是否可用：版本/文档哈希/TFIDF参数/文件完整性"""
        meta = self.load_meta()
        if meta is None:
            return False
        current_hash = self.get_docs_md5_hash()
        if (meta["version"] != self.cache_version
                or meta["doc_hash"] != current_hash
                or meta["tfidf_config"] != self.tfidf_config):
            return False
        required_files = [
            self.chunk_cache_path,
            self.tfidf_model_path,
            self.tfidf_matrix_path
        ]
        for f in required_files:
            if not f.exists():
                return False
        return True

    def load_all_cache(self) -> bool:
        """加载分块+TFIDF模型+向量矩阵缓存"""
        if not self.is_cache_valid():
            print("缓存失效：文档修改/版本升级/参数变更，将重新构建全量索引")
            return False
        try:
            with open(self.chunk_cache_path, "rb") as f:
                self.doc_chunks = pickle.load(f)
            with open(self.tfidf_model_path, "rb") as f:
                self.tfidf_model = pickle.load(f)
            with open(self.tfidf_matrix_path, "rb") as f:
                self.tfidf_matrix = pickle.load(f)
            print(f"✅ 缓存加载成功，共 {len(self.doc_chunks)} 个文本块")
            return True
        except Exception as e:
            print(f"缓存文件损坏 [{e}]，自动清理重建索引")
            self.clear_all_cache()
            return False

    def save_all_cache(self, doc_hash: str):
        """持久化全部缓存"""
        # 保存分块
        try:
            with open(self.chunk_cache_path, "wb") as f:
                pickle.dump(self.doc_chunks, f)
        except Exception as e:
            print(f"文档块缓存保存失败：{e}")
        # 保存TFIDF
        try:
            with open(self.tfidf_model_path, "wb") as f:
                pickle.dump(self.tfidf_model, f)
            with open(self.tfidf_matrix_path, "wb") as f:
                pickle.dump(self.tfidf_matrix, f)
        except Exception as e:
            print(f"TF-IDF向量缓存保存失败：{e}")
        self.save_meta(doc_hash)
        print("✅ 分层缓存持久化完成（文档块 + TF-IDF向量索引）")

    def initialize(self, chunk_size: int = 500):
        """系统初始化入口：缓存优先，无缓存则全量构建"""
        print("===== 制造业SOP智能检索系统初始化 =====")
        self.docs_dir.mkdir(exist_ok=True)
        cache_ok = self.load_all_cache()
        if not cache_ok:
            self.load_documents()
            self.preprocess_documents(chunk_size=chunk_size)
            self.build_vector_index()
            current_hash = self.get_docs_md5_hash()
            self.save_all_cache(current_hash)
        print(f"\n初始化完成，文本块总数：{len(self.doc_chunks)}")

    def load_documents(self):
        """读取docs目录下全部txt文档（新增打印测试日志）"""
        print("\n==== 文档加载测试 ====")
        self.documents.clear()
        all_txt_files = list(self.docs_dir.glob("*.txt"))
        print(f"检测到docs文件夹下txt文件总量：{len(all_txt_files)}")
        print(f"文件列表：{[x.name for x in all_txt_files]}")

        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.documents.append({
                            'file_name': file_path.name,
                            'content': content,
                            'path': str(file_path),
                        })
                        print(f"✅ 成功读取有效文档：{file_path.name}")
                    else:
                        print(f"⚠️ 文件为空已跳过：{file_path.name}")
            except Exception as e:
                print(f"❌ 加载文件 {file_path.name} 失败: {e}")

        print(f"\n【最终有效加载文档数量】：{len(self.documents)}")
        print("========================\n")

    def preprocess_documents(self, chunk_size: int = 500, overlap: int = 100):
        """遍历所有文档执行分块，存入self.doc_chunks"""
        self.doc_chunks.clear()
        chunk_id = 0
        for doc in self.documents:
            chunks = self.split_text_chunks(doc["content"], chunk_size=chunk_size, overlap=overlap)
            for c in chunks:
                self.doc_chunks.append({
                    "chunk_id": chunk_id,
                    "source": doc["file_name"],
                    "full_path": doc["path"],
                    "content": c
                })
                chunk_id += 1
        print(f"文档分块完成，总块数：{len(self.doc_chunks)}")

    def split_text_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """
        分句切割文本块，带重叠窗口
        :param text: 原文
        :param chunk_size: 单块最大字符
        :param overlap: 块间重叠字符
        :return: 分块文本列表
        """
        # 按空行拆分段落
        paragraphs = re.split(r'\n\s*\n', text.strip())
        full_sentences = []
        split_pattern = re.compile(r'([。.!？?；;])')
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            parts = split_pattern.split(para)
            for i in range(0, len(parts), 2):
                sent_text = parts[i].strip()
                if i + 1 < len(parts):
                    sent_text += parts[i + 1]
                if sent_text:
                    full_sentences.append(sent_text)
        # 滑动窗口分块
        chunks = []
        current_chunk = ""
        for sent in full_sentences:
            if len(current_chunk) + len(sent) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 重叠逻辑
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + sent
                else:
                    current_chunk = sent
            else:
                current_chunk += sent
        # 剩余末尾文本
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def chinese_tokenizer(self, text: str):
        """中文分词过滤：去除符号、纯数字、单字"""
        words = jieba.lcut(text)
        filter_words = []
        symbol_filter = {"、", "。", "《", "》", "①", "②", "③", "④", "⑤", "±", "≤", "mm", "℃", "“", "”"}
        for w in words:
            w = w.strip()
            if len(w) > 1 and not w.isdigit() and w not in symbol_filter:
                filter_words.append(w)
        return filter_words

    def build_vector_index(self):
        """训练TF-IDF模型，生成全局向量矩阵"""
        print("正在训练TF-IDF向量索引...")
        train_texts = [chunk["content"] for chunk in self.doc_chunks]
        if not train_texts:
            print("警告：无文本块，无法构建向量索引")
            return
        self.tfidf_model = TfidfVectorizer(
            tokenizer=self.chinese_tokenizer,
            stop_words=self.stop_words,
            max_features=self.tfidf_config["max_features"],
            min_df=self.tfidf_config["min_df"],
            max_df=self.tfidf_config["max_df"],
            ngram_range=self.tfidf_config["ngram_range"],
            sublinear_tf=self.tfidf_config["sublinear_tf"]
        )
        self.tfidf_matrix = self.tfidf_model.fit_transform(train_texts)
        print(f"TF-IDF训练完成，特征维度：{self.tfidf_matrix.shape[1]}")

        # 测试样例查询
        test_queries = [
            "车床开机要检查哪些润滑和防护？",
            "首件检验的操作步骤是什么",
            "金属零件外观哪些缺陷判定不合格",
            "车间机床有哪些禁止操作行为",
            "钢材原材料仓库保管要求"
        ]
        test_vec = self.tfidf_model.transform(test_queries)
        print("测试问句向量形状：", test_vec.shape)

    def search_chunks(self, query: str, top_k: int = 10, similarities_threshold: float = 0.01):
        """
        统一检索入口：先全量过滤阈值，再取top_k相似度片段
        :param query: 用户问题
        :param top_k: 最多返回条数
        :param similarities_threshold: 最低相似度阈值
        :return: 结构化检索结果列表
        """
        query = query.strip()
        if not query:
            return []
        if self.tfidf_model is None or self.tfidf_matrix is None:
            raise ValueError("向量索引未初始化，请先执行initialize()")

        # 问句向量化
        query_vector = self.tfidf_model.transform([query])
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # 1. 筛选超过阈值的索引
        valid_mask = similarities >= similarities_threshold
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) == 0:
            return []

        # 2. 相似度降序排序
        valid_sim = similarities[valid_idx]
        sort_desc_idx = valid_idx[np.argsort(-valid_sim)]

        # 3. 截取前top_k
        top_idx = sort_desc_idx[:top_k]

        res = []
        for idx in top_idx:
            score = round(float(similarities[idx]), 4)
            chunk = self.doc_chunks[idx]
            res.append({
                "相似度": score,
                "文本片段": chunk["content"],
                "来源": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "文件路径": chunk["full_path"]
            })
        return res

    def ask(self, question: str):
        """
        问答对外统一接口
        :param question: 用户提问
        :return: dict 包含检索片段与LLM应答
        """
        # 1. 先检索文本块
        chunks = self.search_chunks(question, top_k=2, similarities_threshold=0.03)
        if not chunks:
            return {
                'question': question,
                'answer': '抱歉在文档中没有找到与您问题相关的规范内容',
                'sources': [],
                'retrieve_chunks': []
            }
        print(f"找到 {len(chunks)} 个相关文档片段")

        # 2. 拼接检索上下文
        context = "\n\n=====文档分割=====\n\n".join([f"【文档：{item['来源']}】\n{item['文本片段']}" for item in chunks])
        # 限制上下文长度防止超窗口
        context = context[:3500]

        # 3. 系统提示词
        SYSTEM_PROMPT = """
        # 角色与目标
        你是由[公司名称]部署的“智能制造知识助手”。你的核心用户是一线生产管理者与车间操作员。你的唯一目标是：基于提供的检索上下文，提供绝对准确、安全合规、易于执行的生产指导。

        # 核心原则（最高优先级）
        1. 安全红线：涉及“车间安全”、“设备急停”、“高压/高温操作”等问题时，必须优先输出安全警告。若检索内容存在冲突，以最新的安全规范为准。
        2. 严格基于上下文：只回答检索到的文档内容。严禁使用外部常识或编造参数（如温度、压力、扭矩值）。若上下文中无答案，直接回复：“当前知识库未收录该问题的具体操作规范，请联系[具体部门/岗位]确认。”
        3. 拒绝模糊：禁止使用“大概”、“可能”、“建议参考”等词汇。必须引用具体数值、标准号或文档名称。

        # 限制条件（Constraints）
        1. 严禁幻觉与编造：绝对禁止使用模型预训练的外部知识。若上下文中没有相关信息，严禁自行推导。
        2. 禁止越权建议：你的角色是“知识检索与传达”，严禁在回答中主动建议修改SOP、降低质检标准或放宽安全限制。若用户反馈现有标准不合理，仅可回复：“已记录您的反馈，请联系工艺/质量工程师进行评估。”
        3. 禁止输出冗长理论：禁止输出大段的背景介绍、原理分析或学术性解释。回答必须直奔主题，直接给出“操作指令”、“排查步骤”或“判定标准”。
        4. 禁止未经确认的替代方案：若用户询问的物料或工具在上下文中未提及，严禁自行推荐其他看似相似的物料或工具作为替代。必须要求用户核实BOM表或联系物料员。
        5. 语言与称呼：统一使用简体中文。禁止使用过于书面化或文言化的词汇。统一称呼用户为“您”或“同事”，保持专业、平等的语气。

        # 多轮对话与追问引导机制
        1. 信息缺失主动追问：当用户提问过于简略（如仅包含设备名、现象或物料名，缺少关键参数、批次号或具体工步）时，严禁盲目猜测。必须列出 2-3 个最可能的排查方向，并引导用户补充关键信息。
        2. 追问话术规范：采用“确认现状 + 给出选项 + 引导回复”的结构。语气保持专业、耐心，降低一线员工的认知负担。
        3. 上下文记忆与继承：在多轮对话中，必须继承上一轮的“设备型号”、“物料编码”或“工序环节”。若用户后续提问省略了主语，系统需自动补全，避免反复询问相同信息。
        4. 死循环熔断：若连续 2 次追问后，用户仍无法提供有效信息或表示“不知道”，系统应主动终止追问，并给出兜底方案：“当前信息暂无法精准定位问题。建议您直接联系[对应岗位/班组长姓名]进行现场确认，或查阅《异常上报流程》。”

        # 回答规范
        1. SOP/操作步骤类：必须使用有序列表。格式：步骤 + 动作 + 关键参数 + 预期结果。
        2. 设备维保类：必须包含：故障现象 -> 排查步骤 -> 解决方案 -> 复位/测试要求。
        3. 质检标准类：必须明确：检验项目 + 公差/标准值 + 判定规则（合格/不合格/让步接收）。
        4. 仓储/物料类：必须包含：物料编码/名称 + 库位 + 先进先出规则 + 有效期/批次要求。

        # 格式与语气
        1. 语气：专业、冷静、指令化。像一位经验丰富的班组长在说话。
        2. 排版：关键参数必须加粗；危险操作使用 ⚠️ 前缀；复杂流程使用表格或分步列表。
        3. 长度：除非用户要求“详细解释”，否则默认提供“执行摘要”，控制在300字以内。

        # 异常处理
        1. 上下文冲突：若检索到多个版本的SOP，优先采用发布日期最近的，并注明：“⚠️ 检测到多版本，以下基于最新文档回答。”
        2. 多片段拼接处理：你接收到的上下文是由多个文档片段拼接而成的。请综合所有片段进行推理，若不同片段间存在信息差异，请以带有最新日期的文档为准；若无法判断新旧，请同时列出差异并提示用户核实。
        3. 话题切换：若用户在讨论设备维保时突然询问仓储物料问题，需平滑切换，并自动清空上一轮的特定设备上下文，避免参数混淆。
        4. 非生产问题：若用户询问薪资、人事等非生产问题，回复：“我是生产知识助手，该问题请咨询HR部门。”

        # 典型场景追问示例（Few-Shot）
        - 用户：“注塑机报警了。”
          助手：“收到。为了准确排查，请确认是以下哪种情况：
          1. 屏幕显示‘加热圈温度异常’？
          2. 屏幕显示‘液压系统压力过低’？
          3. 或者是机械臂动作卡死？
          请回复数字或补充具体的报警代码，我将为您提供对应的处理SOP。”
        - 用户：“这批外壳有划痕。”
          助手：“了解。关于外壳划痕的判定，请问：
          1. 划痕位于哪个区域？（A面/B面/非外观面）
          2. 大概的长度和深度是多少？
          3. 涉及的具体物料批次号是多少？
          补充以上信息后，我将为您匹配最新的《外观质检标准》进行合格判定。”

        # 任务执行
        现在，请严格基于下面【检索上下文】回答用户问题。如果上下文中没有答案，请拒绝回答。
        """.strip()

        # 组装用户输入Prompt，填充上下文和问题
        user_prompt = f"""
        <检索上下文>
        {context}
        </检索上下文>

        用户问题：{question}
        """.strip()

        llm_answer = ""
        try:
            print("==================== 开始发起DeepSeek大模型API请求 ====================")
            # 调用DeepSeek兼容OpenAI接口
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                stream=False,
                temperature=0.7
            )
            print("==================== ✅ 大模型API调用成功 ====================")
            llm_answer = response.choices[0].message.content.strip()
            print(f"大模型原始返回内容：\n{llm_answer}")
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e) if str(e) else "未知错误"
            print("==================== ❌ 大模型API调用失败 ====================")
            print(f"错误类型：{err_type}")
            print(f"错误详情：{err_msg}")
            llm_answer = f"系统调用知识库问答接口异常：{err_type} - {err_msg}"

        # 统一封装返回结果

        return {
            'question': question,
            'answer': llm_answer,
            'sources': list({item['来源'] for item in chunks}),
            'retrieve_chunks': chunks
        }


def main():
    print("=== 制造业SOP智能检索问答系统【控制台测试版】===")
    try:
        # 替换为你的真实DeepSeek Key
        rag = RagSystem(api_key='sk-02b869429d8c435a94f549a0e5cff342', docs_dir='docs')
        rag.initialize()
        # 单独打印文档数量测试
        print(f"\n【测试输出】加载文档总数：{len(rag.documents)}")
        print(f"【测试输出】文本块总数：{len(rag.doc_chunks)}")
    except Exception as e:
        print(f"系统初始化失败：{e}")
        return

    print("\n使用说明：输入问题检索文档，输入 quit / exit / 退出 结束程序")
    while True:
        try:
            question = input("\n请输入你的问题：").strip()
            if question.lower() in ["quit", "exit", "退出"]:
                print("感谢使用，程序退出！")
                break
            if not question:
                print("输入不能为空，请重新输入")
                continue
            result = rag.ask(question)
            print("\n==== 检索返回结果 ====")
            print(f"系统应答：{result['answer']}")
            print(f"关联文档来源：{result['sources']}")
            for idx, item in enumerate(result['retrieve_chunks'], 1):
                print(f"\n[{idx}] 相似度：{item['相似度']} | 文档：{item['来源']}")
                print(f"片段预览：{item['文本片段'][:300]}...")
        except KeyboardInterrupt:
            print("\n检测到中断，程序退出")
            break
        except Exception as e:
            print(f"问答异常：{e}")

if __name__ == '__main__':
    main()