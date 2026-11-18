# rev1 혼합 조건 점검

**결과: 모두 통과**
| 데이터셋 | train 장수 | validation 장수 | train 밀도별 장수 (2·20·50·80 %) | validation 밀도별 장수 | train 점유율 중앙값 (%) |
|---|---:|---:|---|---|---:|
| f1_mini_cmix_s1 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.2 |
| f1_mini_cmix_s2 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.0 |
| f2_air_cmix_s1 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.2 |
| f2_air_cmix_s2 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 21.8 |
| f3_mavic_cmix_s1 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.2 |
| f3_mavic_cmix_s2 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 21.7 |
| f4_pro_cmix_s1 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.4 |
| f4_pro_cmix_s2 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 21.7 |
| f5_other_cmix_s1 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 22.4 |
| f5_other_cmix_s2 | 6000 | 600 | 1500·1500·1500·1500 | 150·150·150·150 | 21.7 |

## f1_mini_cmix_s1

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f1_mini_cmix_s2

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f2_air_cmix_s1

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f2_air_cmix_s2

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f3_mavic_cmix_s1

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f3_mavic_cmix_s2

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f4_pro_cmix_s1

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f4_pro_cmix_s2

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f5_other_cmix_s1

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)

## f5_other_cmix_s2

- 통과: 장수와 밀도별 장수(1:1:1:1)
- 통과: train·validation 에 테스트 fold 기종 없음
- 통과: Neo 2 드론 신호 세그먼트 없음
- 통과: train·validation 에 함께 들어간 IQ 파일 0개
- 통과: 테스트 1 의 IQ 파일이 train·validation 에 0개
- 통과: 배경 노이즈 train Park02_1, validation Park02_2
- 통과: 테스트 이미지와 같은 난수 씨앗 없음
- 통과: 목록 train 6000장, validation 600장, 겹침 0장
- 통과: 목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴
- 통과: 링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)
