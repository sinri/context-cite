import csv
from typing import List, Dict

from transformers import pipeline, GenerationConfig

from context_cite.black_box import BlackBoxCitationAnalyzer, BlackBoxCitationResult


class LLM:
    def __init__(self, model_source):
        self.__pipe = pipeline("text-generation", model=model_source)
        self.__config = GenerationConfig(
            max_new_tokens=1024,
            repetition_penalty=1.5,
        )

    def generate(self, messages: List[Dict[str, str]], remove_think=True):
        response = self.__pipe(messages, generation_config=self.__config,)
        res = response[0]["generated_text"][1]["content"]
        if remove_think:
            if "</think>" in res:
                res = res.split("</think>")[-1]
            res = res.lstrip()
        return res

# model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
model_source = "E:\\sinri\\HuggingFace\\Qwen3-1.7B"
llm = LLM(model_source)

class ReportCsvHelper:
    def __init__(self, target_csv_file_path:str,query:str,target_text:str):
        self.__target_csv_file_path = target_csv_file_path
        self.__query = query
        self.__target_text = target_text
        self.__records=[]

    def add_exp_round_record(self,round:int,output_1:str,output_2:str,influence_1:str,influence_2:str):
        self.__records.append({
            'round':round,
            'output_1':output_1,
            'output_2':output_2,
            'influence_1':influence_1,
            'influence_2':influence_2,
        })

    def write(self):
        with open(self.__target_csv_file_path, 'w',encoding='gbk') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['target_text:',self.__target_text])
            csvwriter.writerow(['query:', self.__query])
            csvwriter.writerow(['round','output_1','output_2','influence_1','influence_2'])
            for record in self.__records:
                csvwriter.writerow([record['round'],record['output_1'],record['output_2'],record['influence_1'],record['influence_2']])



def single(llm, query, target_text, output_1, output_2):
    analyzer = BlackBoxCitationAnalyzer(model_source=model_source)
    result = analyzer.analyze(
        query=query,
        target_text=target_text,
        output_1=output_1,
        output_2=output_2,
    )

    influence_1 = result.get_influence_on_output_1()
    influence_2 = result.get_influence_on_output_2()

    # 数值越大，表示在本地模型 m 的视角下，这段回答越依赖 target_text；
    # 通常期望 influence_on_output_2 ≥ influence_on_output_1（因为 output_2 是在含 target_text 的上下文中生成的）。
    # > 0	target_text 作为 context 时，提高了模型生成该段 response 的“得分”（与“被引用”一致）
    # < 0	target_text 降低了该段 response 的得分（与引用相悖）
    # ≈ 0	几乎无影响

    print('output 1')
    print(output_1)
    print('output 2')
    print(output_2)
    print(influence_1, influence_2)


def repeat_exp(llm:LLM, query, target_text, repeat=5,report_file_path:str=None):
    sum1 = 0
    sum2 = 0
    for round in range(repeat):
        print(f'Round [{round + 1}/{repeat}] start...')
        output_1 = llm.generate(messages=[{"role": "user", "content": "Query: \n"+query}], )
        output_2 = llm.generate(messages=[{"role": "user", "content": "Reference maybe useful: \n" + target_text + "\n\nQuery: \n" + query}], )

        print(f"output 1: {output_1}")
        print(f"output 2: {output_2}")

        analyzer = BlackBoxCitationAnalyzer(model_source=model_source)
        result = analyzer.analyze(
            query=query,
            target_text=target_text,
            output_1=output_1,
            output_2=output_2,
        )
        print(
            f'Round [{round + 1}/{repeat}]: {result.get_influence_on_output_1()} vs {result.get_influence_on_output_2()}')
        sum1 += result.get_influence_on_output_1()
        sum2 += result.get_influence_on_output_2()

    fin_result = BlackBoxCitationResult(influence_on_output_1=sum1 / repeat, influence_on_output_2=sum2 / repeat)

    print(f"Average: {fin_result.get_influence_on_output_1()} vs {fin_result.get_influence_on_output_2()}")
    print(fin_result.get_conclusion())

def test_1(llm):
    query = "What is the capital of Bioland?"
    target_text = "Findois is the capital and largest city of Bioland."
    single(llm, query, target_text, output_1="I'm not sure.", output_2="Findois is the capital of Bioland.")

def test_2(llm):
    query = "中华人民共和国的首都是什么"
    target_text = "中国（PRC）的首都自1949年建国起一直是北京。"

    repeat_exp(llm, query, target_text, repeat=5)

def test_3(llm):
    query_1 = "乐其电商的技术团队分布在哪些城市？"
    target_text="""
    乐其电商创立于2009年，是全价值链品牌数字及电商服务伙伴，员工超4000人。
    通过12大城市(杭州、 上海、 北京、 广州、 东莞、 盐城、 海南、 胡志明市、 香港、 东京、 首尔、 新加坡)中设立的据点与世界同频。
    技术团队总部在杭州，在上海、香港、胡志明市也有技术办公室。
    以 `多维专业能力+数据技术` 双驱引擎， 提供 **全链路** 解决方案。
    """
    query_2 = f"""
    {query_1}
    
    {target_text}
    """

    output_1="""
    我需要查找乐其电商的技术团队分布情况。由于这是一个具体的企业组织信息，我将使用联网搜索功能来获取最新和最准确的信息。

根据目前搜索结果，**未直接披露乐其电商（乐其集团）技术团队的具体城市分布信息**。

不过，综合多条权威信息可推断出其核心技术与业务布局集中在以下城市：

✅ **杭州**：  
- 乐其集团总部所在地。2024年8月，乐其集团与一知智能在**杭州高新区（滨江）互联网小镇**联合成立合资公司“其知KiWise”，并由杭州高新区管委会领导出席启动仪式。  
- 中欧国际工商学院课程中引入的乐其AI+电商案例，也明确指向其在杭州的运营与技术实践（如数字人直播间、AIGC内容生产等），表明杭州是其技术研发与商业化落地的核心枢纽。

✅ **深圳**：  
- 搜索结果高频出现“**深圳市乐其创新股份有限公司**”（SmallRig斯莫格品牌主体），注册地为深圳，且多次获深圳企业创新纪录、广东省制造业单项冠军等认定。  
- 虽然SmallRig属乐其集团旗下独立运营的影像科技板块（偏硬件/制造），但其“快制造”技术中台、AI驱动的敏捷供应链系统、日均1.6款新品的研发节奏，高度依赖强大的软件工程、嵌入式开发与AI算法团队——这些能力与乐其电商的技术体系存在协同与共享基础。  
- 深圳作为中国硬科技与跨境电商重镇，极可能是其**前沿技术（如AIoT、边缘计算、跨境数据平台）研发与出海技术支撑的重要基地**。

⚠️ 其他城市（如北京、上海）虽有合作高校（清华、牛津、中欧）、客户或生态伙伴，但**暂无公开信息证实设有常驻技术团队办公点**；部分运营/商务职能可能分布于一二线城市，但非技术主力。

---

🔍 补充说明：  
- “乐其电商”通常指服务宝洁、联合利华、雅诗兰黛等300+国际品牌的**品牌电商全链路服务商**（含天猫/京东/抖音代运营、直播中台、数字人、AIGC内容工厂等），其技术底座强调“运技一体”，即运营经验与AI工程能力深度融合。  
- 其合资子公司“其知KiWise”注册地址及主要办公地仍在**杭州**，进一步强化杭州作为AI电商技术研发中心的地位。

---

✅ 结论：  
截至目前公开信息，**乐其电商的技术团队主要集中于杭州（总部及AI研发中心）和深圳（创新产品与出海技术支撑基地）**，其中杭州为战略与算法/平台研发核心，深圳侧重产品化、供应链智能化与全球化技术适配。

如需进一步确认具体办公地点或团队规模，建议查阅乐其集团官网“加入我们”页面或天眼查/企查查中的分支机构信息。是否需要我帮你检索其官网招聘页或工商注册分支机构？
    """
    output_2="""
    根据您提供的信息，乐其电商的技术团队分布在以下城市：

- **杭州**（技术团队总部）  
- **上海**  
- **香港**  
- **胡志明市**

✅ 这四地明确被提及为设有技术办公室。

（注：虽然公司在全国及海外共12个城市设有据点，但仅上述4个被特别标注为“技术办公室”。）

如需进一步验证最新布局（例如是否有新增技术据点），我可为您联网搜索确认。是否需要？
    """

    single(llm, query_1, target_text, output_1=output_1, output_2=output_2)

def test_4(llm):
    """
    本实验的目的是验证 query 与 target_text 低相关度的情况下，归因分数分布的情况。
    """
    query = "What is the capital of Russia?"
    target_text = "Findois is the capital and largest city of Bioland."

    repeat_exp(llm, query, target_text, repeat=5)


if __name__ == '__main__':
    test_4(llm)
