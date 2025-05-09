# LangGraph Agent 리팩토링 가이드

이 문서는 LangGraph Agent 프로젝트의 리팩토링 가이드를 제공합니다. LangChain과 LangGraph의 최신 아키텍처와 모범 사례를 따라 프로젝트를 재구성하는 방법을 설명합니다.

## 1. 아키텍처 개요

LangChain/LangGraph 아키텍처는 다음과 같은 세 계층으로 구성됩니다:

1. **도구(Tools)**: 개별 작업을 수행하는 특정 기능
2. **에이전트(Agents)**: 도구를 사용하여 도메인별 문제를 해결하는 도메인 전문가
3. **슈퍼바이저(Supervisor)**: 복잡한 작업을 해결하기 위해 에이전트를 조율하는 LangGraph

## 2. 도구(Tools) 구현 가이드

도구는 에이전트가 사용할 수 있는 개별 기능입니다. LangChain의 `@tool` 데코레이터를 사용하여 구현합니다.

### 도구 구현 원칙

1. **단일 책임**: 각 도구는 하나의 명확한 기능을 수행해야 합니다.
2. **명확한 설명**: 도구의 이름과 설명은 에이전트가 이해하기 쉽게 작성해야 합니다.
3. **입력 스키마**: 도구의 입력 매개변수는 명확하게 정의되어야 합니다.
4. **오류 처리**: 도구는 오류를 적절히 처리하고 유용한 오류 메시지를 반환해야 합니다.

### 도구 구현 예시

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """
    웹에서 정보를 검색합니다.
    
    Args:
        query: 검색할 쿼리
        
    Returns:
        검색 결과
    """
    # 검색 로직 구현
    return "검색 결과"
```

### 사용할 LangChain 도구

LangChain은 다양한 내장 도구를 제공합니다. 프로젝트에서 사용할 주요 도구는 다음과 같습니다:

1. **검색 도구**:
   - `TavilySearchTool`: 웹 검색을 위한 도구
   - `GoogleSerperTool`: Google 검색을 위한 도구

2. **이미지 생성 도구**:
   - `DallEImageGeneratorTool`: DALL-E를 사용한 이미지 생성 도구

3. **데이터베이스 도구**:
   - `SQLDatabaseTool`: SQL 데이터베이스 쿼리 도구
   - `VectorStoreTool`: 벡터 저장소 쿼리 도구

4. **유틸리티 도구**:
   - `PythonREPLTool`: Python 코드 실행 도구
   - `RequestsGetTool`: HTTP 요청 도구
   - `FileSystemTool`: 파일 시스템 조작 도구

## 3. 에이전트(Agents) 구현 가이드

에이전트는 도메인 전문가로서 특정 영역의 문제를 해결하는 데 특화된 LangGraph 구성입니다. LangGraph의 `create_react_agent` 함수를 사용하여 구현합니다.

### 에이전트 구현 원칙

1. **도메인 전문성**: 각 에이전트는 특정 도메인의 전문가로 설계되어야 합니다.
2. **적절한 도구 선택**: 도메인에 적합한 도구들을 선택하고 활용해야 합니다.
3. **명확한 시스템 프롬프트**: 에이전트의 시스템 프롬프트는 해당 도메인의 전문가처럼 행동하도록 설계되어야 합니다.
4. **효과적인 추론**: 에이전트는 ReAct 패턴(Reason, Act, Observe)을 따라 효과적으로 추론해야 합니다.

### 에이전트 구현 예시

```python
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

# 도구 정의
tools = [search_web]

# 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 웹 검색 및 정보 검색 전문가입니다. 사용자의 질문을 이해하고, 가장 관련성 높은 정보를 찾기 위한 최적의 검색 전략을 수립하며, 검색 결과의 신뢰성과 관련성을 평가할 수 있습니다."),
    ("user", "{input}"),
    ("user", "중간 단계: {intermediate_steps}")
])

# 에이전트 생성
search_agent = create_react_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4.1"),
    tools=tools,
    prompt=prompt
)
```

### 구현할 도메인 전문가 에이전트

1. **검색 전문가 에이전트**:
   - 도메인: 웹 검색 및 정보 검색
   - 도구: TavilySearchTool, GoogleSerperTool
   - 시스템 프롬프트: "당신은 웹 검색 및 정보 검색 전문가입니다. 사용자의 질문을 이해하고, 가장 관련성 높은 정보를 찾기 위한 최적의 검색 전략을 수립하며, 검색 결과의 신뢰성과 관련성을 평가할 수 있습니다."

2. **이미지 생성 전문가 에이전트**:
   - 도메인: 텍스트-이미지 변환 및 시각적 콘텐츠 생성
   - 도구: DallEImageGeneratorTool
   - 시스템 프롬프트: "당신은 텍스트 설명을 시각적 이미지로 변환하는 전문가입니다. 사용자의 요구사항을 이해하고, 최적의 이미지 생성 프롬프트를 작성하며, 다양한 스타일과 품질 옵션을 활용하여 사용자의 비전을 실현할 수 있습니다."

3. **보고서 작성 전문가 에이전트**:
   - 도메인: 전문적인 보고서 작성 및 문서 구성
   - 도구: FileSystemTool, TavilySearchTool
   - 시스템 프롬프트: "당신은 전문적인 보고서 작성 전문가입니다. 복잡한 정보를 명확하고 구조화된 보고서로 조직화하고, 데이터를 효과적으로 분석하며, 전문적인 형식과 표준을 준수하는 고품질 문서를 생성할 수 있습니다."

4. **데이터베이스 전문가 에이전트**:
   - 도메인: SQL 데이터베이스 쿼리 및 데이터 분석
   - 도구: SQLDatabaseTool, PythonREPLTool
   - 시스템 프롬프트: "당신은 데이터베이스 및 SQL 쿼리 전문가입니다. 복잡한 데이터베이스 스키마를 이해하고, 효율적인 SQL 쿼리를 작성하며, 데이터를 의미 있는 인사이트로 변환할 수 있습니다."

5. **지식 관리 전문가 에이전트**:
   - 도메인: 벡터 데이터베이스 및 지식 검색
   - 도구: VectorStoreTool
   - 시스템 프롬프트: "당신은 지식 관리 및 검색 전문가입니다. 정보를 효과적으로 저장하고, 의미적 유사성을 기반으로 관련 지식을 검색하며, 복잡한 지식 구조를 구성하고 관리할 수 있습니다."

## 4. 슈퍼바이저(Supervisor) 구현 가이드

슈퍼바이저는 여러 에이전트를 조율하여 복잡한 작업을 해결하는 LangGraph입니다. LangGraph의 `StateGraph`를 사용하여 구현합니다.

### 슈퍼바이저 구현 원칙

1. **작업 분석**: 사용자의 요청을 분석하여 필요한 에이전트를 결정합니다.
2. **작업 분해**: 복잡한 작업을 여러 에이전트가 협력할 수 있는 하위 작업으로 분해합니다.
3. **에이전트 조율**: 여러 에이전트 간의 작업 흐름을 조율합니다.
4. **결과 통합**: 여러 에이전트의 결과를 일관된 응답으로 통합합니다.

### 슈퍼바이저 구현 예시

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 에이전트 노드 생성
search_agent_node = ToolNode(search_agent)
image_agent_node = ToolNode(image_agent)

# 슈퍼바이저 에이전트 생성
supervisor_agent = create_react_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4.1"),
    tools=[
        create_handoff_tool(agent_name="search_agent", description="검색 전문가에게 작업 할당"),
        create_handoff_tool(agent_name="image_agent", description="이미지 생성 전문가에게 작업 할당")
    ],
    prompt=ChatPromptTemplate.from_messages([
        ("system", "당신은 여러 전문가 에이전트를 조율하는 슈퍼바이저입니다. 사용자의 요청을 분석하고, 적절한 전문가에게 작업을 할당하세요."),
        ("user", "{input}"),
        ("user", "중간 단계: {intermediate_steps}")
    ])
)

# 슈퍼바이저 그래프 생성
supervisor_graph = (
    StateGraph()
    .add_node("supervisor", supervisor_agent)
    .add_node("search_agent", search_agent_node)
    .add_node("image_agent", image_agent_node)
    .add_edge(START, "supervisor")
    .add_edge("search_agent", "supervisor")
    .add_edge("image_agent", "supervisor")
    .add_edge("supervisor", END)
    .compile()
)
```

## 5. 리팩토링 단계

1. **도구 구현**:
   - 기본 도구 인터페이스 생성
   - LangChain 내장 도구 통합
   - 커스텀 도구 구현

2. **도메인 전문가 에이전트 구현**:
   - 각 도메인별 에이전트 구현
   - 도메인 특화 시스템 프롬프트 개발
   - 도메인에 적합한 도구 조합 구성

3. **슈퍼바이저 구현**:
   - 에이전트 선택 로직 구현
   - 작업 분해 및 조율 기능 구현
   - 결과 통합 및 품질 관리 메커니즘 구현

4. **API 및 UI 업데이트**:
   - API 엔드포인트 업데이트
   - UI 컴포넌트 업데이트

## 6. 참고 자료

- [LangChain 도구 문서](https://python.langchain.com/docs/concepts/tools/)
- [LangChain 도구 통합](https://python.langchain.com/docs/integrations/tools/)
- [LangGraph 다중 에이전트 슈퍼바이저 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [LangChain 에이전트 개념](https://python.langchain.com/v0.1/docs/modules/agents/concepts/)
