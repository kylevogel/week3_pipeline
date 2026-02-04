# College Completion Dataset

# 1) Are higher graduation rates associated with higher levels of funding per student?
# 2) I believe the best Independent Business Metric for this problem would be the perecent of pell grant recipients as a measurement of student socioeconomic status
# ^ I'm curious to see if higher graduation rates are related to lower pell recipients per university.
# 3) My instincts tell me the data is heavily numerical consisting of tons of categories. This makes me thin whether or not I can narrow down this data.
# ^ I do see some missing data which I'm rather worried about, but overall I think this set can asnwer my question with great detail.

# %%
# Exploring and getting a feel for the data and its details.
import pandas as pd
df =  pd.read_csv("Data/cc_institution_details.csv")
df.head()
df.info()
df.shape
# Upon further observation, I see a heavy amount of non-null and strings
# The rest will be moved to question_4.py




# Job Placement Dataset

# 1) How do academic and demographic factors whether a student will be placed or not
# 2) A good IBM would be the placement rate of students and % of academic/demographic diversity 
# 3) My instincts tell me that the data is fairly numerical. However, I am worried about the missing data in certain columns.
# %%
# Exploring the dataset
import pandas as pd
pd.set_option("display.max_rows", None)
df2 = pd.read_csv("Data/job_placement.csv")
df2.head()
df2.info()
# Above half are strings...hmm
