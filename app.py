import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. 페이지 설정 (브라우저 탭 이름 및 아이콘)
st.set_page_config(
    page_title="이미지 분류기 Mission",
    page_icon="🖼️",
    layout="wide"
)

# 2. 제목 및 설명
st.title("🖼️ 이미지 분류 AI")
st.markdown("""
이 서비스는 **Google의 ViT(Vision Transformer)** 모델을 사용하여 이미지를 분류합니다.  
이미지를 업로드하고 **'분류하기'** 버튼을 눌러보세요.
""")
st.divider() # 구분선

# 3. 모델 로딩 함수 (캐싱 적용)
# @st.cache_resource: 모델을 전역 메모리에 한 번만 로드하여 속도를 높임
@st.cache_resource
def load_classifier():
    # Hugging Face의 pipeline을 사용하여 모델 로드
    # model: google/vit-base-patch16-224 (이미지넷 1000개 클래스 학습)
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    return classifier

# 4. 사이드바 or 메인에 파일 업로더 배치
uploaded_file = st.file_uploader("이미지 파일을 업로드해주세요", type=["jpg", "png", "jpeg", "webp"])

# 파일이 업로드 되었을 때만 실행
if uploaded_file is not None:
    # 이미지를 PIL 객체로 변환
    image = Image.open(uploaded_file)
    
    # 화면을 2분할 (왼쪽: 이미지, 오른쪽: 결과)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("업로드된 이미지")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("분류 결과")
        
        # 버튼을 누르면 예측 시작
        if st.button("🔍 분류하기", type="primary"):
            # 로딩 스피너 표시
            with st.spinner("AI가 이미지를 분석 중입니다..."):
                try:
                    # 모델 로드 및 예측 수행
                    classifier = load_classifier()
                    # top_k=3: 상위 3개 예측 결과만 가져옴
                    results = classifier(image, top_k=3)
                    
                    # 가장 높은 확률의 결과 강조 표시
                    top_result = results[0]
                    label = top_result['label']
                    score = top_result['score']
                    
                    st.success(f"이 이미지는 **[{label}]** 일 확률이 높습니다! ({score*100:.1f}%)")
                    
                    # 상위 3개 결과 시각화 (Progress bar)
                    st.markdown("---")
                    st.write("**상세 분석 결과:**")
                    
                    for res in results:
                        res_label = res['label']
                        res_score = res['score']
                        
                        # 텍스트와 프로그레스 바 출력
                        st.markdown(f"**{res_label}** ({res_score*100:.1f}%)")
                        st.progress(res_score)
                        
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")