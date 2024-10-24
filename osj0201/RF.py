import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic.head(5))
# 생존(Survived): 0 = 죽은사람, 1 = 살아남은사람
# 객실등급(pclass): 1등급 = 1, 2등급 = 2, 3등급 = 3
# 성별(sex): male = 남자, female = 여자
# 나이(age)




print(titanic.describe())