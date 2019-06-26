# WeatherContest
For 2019 Weather BigData Contest



날씨 데이터 (기상청 제공)
SNS 데이터 (트위터 제공)

Topic : 날씨에 따라서 먹고 싶은 음식, 먹는 음식의 경향성 상관관계

<br>

## Members & Contribution

<br>

| 이름 <br>(Name) | 소속 | 최근에 한 일 <br>(log : Work Update) |
|---|:---:|:---:|
| 최윤영 | 이화여자대학교 / 통계학3 | 6/24 : 크롤러 발견 |
| 장문경 | 세종대학교 / 컴퓨터공학3 | 6/24 : 앱 디자인 제작 |
| 나영채 | 세종대학교 / 생명공학1 | 6/24 : 트위터 API key 발급 |
| 이장후 | 세종대학교 / 컴퓨터공학2 | 6/24 10am : readme, 회의록 작성 |


<br>

## 제출 개요

<br>

| 목차 | 내용 |
|:---:|:---|
| 활용 방안| 연관성을 보여서, 마케팅 자료로 활용할 수 있는 가능성을 보인다. |
| | |




<br>

## 참고 자료

<br>

> 트위터 크롤링 관련
- [트위터 비공식 크롤링 도구](https://github.com/truptoupparabpara/twitterscraper)
  - [예제코드](https://fouaaa.blogspot.com/2019/01/capstone-design-twitterscraper-python.html)

> 감정분석 관련
- [R을 이용한 감정 분석](http://ruck2015.r-kor.org/handout/sentiment_analysis_hyungjunkim.pdf)
- [마이크로소프트 Azure 텍스트분석](https://docs.microsoft.com/ko-kr/azure/cognitive-services/text-analytics/language-support)
  - [언어지원현황](https://docs.microsoft.com/ko-kr/azure/cognitive-services/text-analytics/language-support)





<br>

## 회의록

<br>

| 날짜 | 내용 | 다음 시간까지 준비해야 하는 내용 |
|:---:|---|---|
| 2019-06-24 9:00AM | 트위터를 이용한 감정 분석을 진행하려 했으나, 날씨는 사람마다 호와 불호가 있다는 점 때문에 유의미한 감정의 상관관계를 알아보기 어렵다는 문제점을 발견. 음식과의 상관관계를 분석하는 것이 더 구체적일 것이라고 판단함. |  |
| 2019-06-24 11:00PM | 그래도 한 번 해보자! 감정분석을 할 수 있는 방법이 없는지 생각해 본다.<br>윤영 : 키워드 없이 추출할 수 있는 도구도 크롤러에서 함께 제공한다. <br>영채 : 이전에 영화 평점 데이터를 label 로 사용해서, 긍정과 부정 데이터를 가져온 적 있다. 하지만, 감정은 긍정부정으로만 나눌 수 있는 것이 아니다. <br>영채 : 마이크로소프트 클라우드에 감정분석 도구를 제공한다.<br>너무 어렵다 갈아엎자! 이건 어떤가? 대한민국 top K개의 카페에 공통적으로 존재하는 메뉴를 잘 묶어서 SNS에 올라오는 빈도수를 분석하는 것. 날씨에 따라 카페에서 어떤 소비경향을 보이는지 분석해 본다. |  |
|2019-06-24 23:00PM |커피와 날씨의 상관관계를 분석하기로 함. 네이버 검색어를 비교해본 결과 날씨가 더웠을때 카라멜 마끼아또가 아메리카노보다 우위에 속했음<br> 인스타 해시태그 크롤러 사용, 네이버 검색어 순위 사용 | |
|2019-07-04 6:00PM | | |
|**2019-07-22** | **날씨 빅데이터 공모전 신청 마감** | |


