import json

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from supabase import Client, create_client


SUMMARY_MODELS = ["gpt-4o-mini", "gpt-5.6-luna"]
QUIZ_MODEL = "gpt-4o-mini"
SUMMARY_SYSTEM_PROMPT = """다음 학습 자료를 분석하고 반드시 JSON 객체 하나만 반환하세요.
JSON 형식:
{
  "핵심 요약": "핵심 내용을 정확히 3문장으로 요약",
  "주요 키워드": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "난이도": "상 또는 중 또는 하"
}
핵심 요약은 정확히 3문장, 주요 키워드는 정확히 5개를 작성하세요.
"""
QUIZ_SYSTEM_PROMPT = """당신은 학습 퀴즈를 만드는 도우미입니다.
주어진 요약을 바탕으로 4지선다 퀴즈 3문항을 만드세요.
각 문항은 question, options(4개), answer(정답 인덱스, 0부터 시작), explanation을 포함해야 합니다.
결과는 {"questions": [...]} 형태의 JSON으로 반환하세요.
"""


def summarize_with_model(model: str, document_text: str) -> dict:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": document_text},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("모델이 빈 응답을 반환했습니다.")
    return json.loads(content)


def create_quiz(summary_results: dict) -> dict:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=QUIZ_MODEL,
        messages=[
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(summary_results, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("모델이 빈 응답을 반환했습니다.")
    return json.loads(content)


def display_summary(summary: dict) -> None:
    st.markdown("**핵심 요약**")
    st.write(summary.get("핵심 요약", "결과 없음"))
    st.markdown("**주요 키워드**")
    keywords = summary.get("주요 키워드", [])
    st.write(", ".join(keywords) if isinstance(keywords, list) else keywords)
    st.markdown("**난이도**")
    st.write(summary.get("난이도", "결과 없음"))


def display_quiz(quiz: dict) -> None:
    questions = quiz.get("questions", [])
    for question_number, question in enumerate(questions, start=1):
        st.markdown(f"#### {question_number}. {question.get('question', '문제 없음')}")
        options = question.get("options", [])
        selected_option = st.radio(
            "보기",
            options,
            index=None,
            key=f"quiz-option-{question_number}",
            label_visibility="collapsed",
        )
        if selected_option is not None:
            selected_index = options.index(selected_option)
            answer_index = int(question.get("answer", -1))
            if selected_index == answer_index:
                st.success("정답입니다.")
            else:
                correct_option = options[answer_index] if 0 <= answer_index < len(options) else "확인할 수 없음"
                st.error(f"오답입니다. 정답: {correct_option}")
            st.info(f"해설: {question.get('explanation', '해설 없음')}")


def get_supabase_client() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def get_summary_preview(summary: dict) -> str:
    for model_summary in summary.values():
        if isinstance(model_summary, dict) and model_summary.get("핵심 요약"):
            preview = str(model_summary["핵심 요약"]).replace("\n", " ")
            return preview[:120] + ("..." if len(preview) > 120 else "")
    return "요약 내용 없음"


def load_saved_summaries() -> list[dict]:
    client = get_supabase_client()
    response = (
        client.table("summaries")
        .select("id, filename, summary, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


def save_summary(filename: str, summary: dict) -> object:
    client = get_supabase_client()
    response = client.table("summaries").insert(
        {"filename": filename, "summary": summary, "quiz": None}
    ).execute()
    if not response.data:
        raise ValueError("저장된 요약의 id를 확인할 수 없습니다.")
    return response.data[0]["id"]


def update_quiz(summary_id: object, quiz: dict) -> None:
    client = get_supabase_client()
    client.table("summaries").update({"quiz": quiz}).eq("id", summary_id).execute()


def delete_summary(summary_id: object) -> None:
    client = get_supabase_client()
    client.table("summaries").delete().eq("id", summary_id).execute()


st.set_page_config(
    page_title="AI 학습 정리 노트",
    page_icon="📚",
    layout="centered",
)

with st.sidebar:
    st.header("문서 업로드")
    uploaded_file = st.file_uploader(
        "학습 자료 PDF 업로드",
        type=["pdf"],
        accept_multiple_files=False,
    )

st.title("AI 학습 정리 노트")
st.caption("PDF를 업로드하면 학습 내용을 바로 확인할 수 있어요.")

if uploaded_file is None:
    original_tab, summary_tab, quiz_tab, history_tab = st.tabs(["원문", "요약", "퀴즈", "저장 기록"])
    with original_tab:
        st.info("PDF 파일을 업로드해 주세요.")
    with summary_tab:
        st.info("요약하려면 먼저 PDF 파일을 업로드해 주세요.")
    with quiz_tab:
        st.info("퀴즈를 만들려면 먼저 PDF를 요약하고 저장해 주세요.")
    with history_tab:
        st.subheader("저장된 요약 기록")
        try:
            saved_summaries = load_saved_summaries()
            if not saved_summaries:
                st.info("저장된 요약 기록이 없습니다.")
            for saved_summary in saved_summaries:
                record_id = saved_summary.get("id")
                record_columns = st.columns([3, 2, 1])
                with record_columns[0]:
                    st.markdown(f"**{saved_summary.get('filename', '파일명 없음')}**")
                    st.caption(get_summary_preview(saved_summary.get("summary", {})))
                with record_columns[1]:
                    st.caption(str(saved_summary.get("created_at", "저장 시각 없음")))
                with record_columns[2]:
                    if st.button("삭제", key=f"delete-summary-{record_id}"):
                        delete_summary(record_id)
                        st.rerun()
        except Exception as error:
            st.error(f"저장 기록을 불러오지 못했습니다: {error}")
else:
    try:
        reader = PdfReader(uploaded_file)
        page_count = len(reader.pages)
        extracted_pages = [page.extract_text() or "" for page in reader.pages]
        document_text = "\n\n".join(extracted_pages).strip()
        document_id = f"{uploaded_file.name}:{uploaded_file.size}"

        with st.sidebar:
            st.divider()
            st.subheader("문서 정보")
            st.text(f"파일명: {uploaded_file.name}")
            st.metric("총 페이지 수", f"{page_count}페이지")

        if st.session_state.get("summary_document_id") != document_id:
            st.session_state.summary_document_id = document_id
            st.session_state.summary_results = {}
            st.session_state.quiz_result = None
            st.session_state.saved_summary_id = None

        summary_results = st.session_state.setdefault("summary_results", {})
        original_tab, summary_tab, quiz_tab, history_tab = st.tabs(["원문", "요약", "퀴즈", "저장 기록"])

        with original_tab:
            st.subheader("페이지별 텍스트")
            if any(page_text.strip() for page_text in extracted_pages):
                for page_number, page_text in enumerate(extracted_pages, start=1):
                    with st.expander(f"{page_number}페이지", expanded=page_number == 1):
                        if page_text.strip():
                            st.text_area(
                                f"{page_number}페이지 텍스트",
                                page_text,
                                height=260,
                                label_visibility="collapsed",
                                key=f"page-text-{page_number}",
                            )
                        else:
                            st.info("이 페이지에서는 추출할 수 있는 텍스트가 없습니다.")
            else:
                st.warning("PDF에서 추출할 수 있는 텍스트가 없습니다. 이미지로 저장된 PDF일 수 있어요.")

        with summary_tab:
            st.subheader("모델별 요약 비교")
            if not document_text:
                st.warning("요약할 수 있는 텍스트가 없습니다.")
            elif st.button("요약하기", type="primary"):
                if "OPENAI_API_KEY" not in st.secrets:
                    st.error("OpenAI API 키가 설정되지 않았습니다. .streamlit/secrets.toml을 확인해 주세요.")
                else:
                    st.session_state.saved_summary_id = None
                    st.session_state.quiz_result = None
                    with st.spinner("두 모델이 문서를 요약하고 있습니다..."):
                        for model in SUMMARY_MODELS:
                            try:
                                summary_results[model] = summarize_with_model(model, document_text)
                            except Exception as error:
                                summary_results[model] = {"error": str(error)}

            if summary_results:
                columns = st.columns(len(SUMMARY_MODELS))
                for column, model in zip(columns, SUMMARY_MODELS):
                    with column:
                        st.markdown(f"#### {model}")
                        result = summary_results.get(model)
                        if result and "error" in result:
                            st.error(f"요약 실패: {result['error']}")
                        elif result:
                            display_summary(result)

                if st.button("저장", key="save-summary", type="secondary"):
                    try:
                        st.session_state.saved_summary_id = save_summary(uploaded_file.name, summary_results)
                        st.success("요약 결과를 저장했습니다.")
                    except Exception as error:
                        st.error(f"요약 결과를 저장하지 못했습니다: {error}")

        with quiz_tab:
            st.subheader("학습 퀴즈")
            quiz_result = st.session_state.get("quiz_result")
            if not summary_results:
                st.info("먼저 요약 탭에서 요약을 만들어 주세요.")
            elif st.session_state.get("saved_summary_id") is None:
                st.info("퀴즈를 만들기 전에 요약 결과를 먼저 저장해주세요.")
            else:
                if st.button("퀴즈 생성", type="primary"):
                    if "OPENAI_API_KEY" not in st.secrets:
                        st.error("OpenAI API 키가 설정되지 않았습니다. .streamlit/secrets.toml을 확인해 주세요.")
                    else:
                        try:
                            with st.spinner("요약을 바탕으로 퀴즈를 만들고 저장하는 중입니다..."):
                                quiz_result = create_quiz(summary_results)
                                update_quiz(st.session_state.saved_summary_id, quiz_result)
                                st.session_state.quiz_result = quiz_result
                            st.success("퀴즈를 만들고 저장했습니다.")
                        except Exception as error:
                            st.error(f"퀴즈를 생성하거나 저장하지 못했습니다: {error}")

                if quiz_result:
                    display_quiz(quiz_result)

        with history_tab:
            st.subheader("저장된 요약 기록")
            try:
                saved_summaries = load_saved_summaries()
                if not saved_summaries:
                    st.info("저장된 요약 기록이 없습니다.")
                for saved_summary in saved_summaries:
                    record_id = saved_summary.get("id")
                    record_columns = st.columns([3, 2, 1])
                    with record_columns[0]:
                        st.markdown(f"**{saved_summary.get('filename', '파일명 없음')}**")
                        st.caption(get_summary_preview(saved_summary.get("summary", {})))
                    with record_columns[1]:
                        st.caption(str(saved_summary.get("created_at", "저장 시각 없음")))
                    with record_columns[2]:
                        if st.button("삭제", key=f"delete-summary-{record_id}"):
                            delete_summary(record_id)
                            st.rerun()
            except Exception as error:
                st.error(f"저장 기록을 불러오지 못했습니다: {error}")
    except Exception:
        st.error("PDF를 읽는 중 문제가 발생했습니다. 파일이 손상됐수~..확인해보슈..")
