import streamlit as st 
from PIL import Image

#title
st.title("Resources")
#page intro
st.write("This page contains helpful resources for guirdains and children to prevent gromming or deal with the aftermath")
#columns to organize the page
col1, col2 = st.columns([2,2])


c1 = st.container(border=False)
c2 = st.container(border=False)

#subtitle of the first section which is about prevention and education
c1.markdown("### Prevention")
#picture var
pic1 = Image.open(r"protection.png")
#display the image 
c1.image(pic1, width=300)
c1.write("**Grooming Takes Time.** Abusers do not just walk up to kids and say, “I want to touch your private parts” because they know the kid will probably say “No!” They want to abuse kids, but they don’t want to scare the kid away too soon, and they don’t want to get caught, so they build invisible “traps,” kind of like spider webs, in the hope that the kid won’t notice until they’ve already been caught. Lots of kids have good background knowledge of prey animals from watching animal shows on TV. I often ask if they can think of an animal who sets up traps. This help them make connections that help build their understanding about grooming.")


#divider 
c1.divider()

c1.write("")
#resources for this section is all articles to educate about grooming
c1.write("**more resources**")
url = "https://www.schoolcounselingbyheart.com/2012/08/26/teaching-kids-to-recognize-grooming/"
c1.write("- [Teaching Kids to Recognize Grooming](%s)" % url)
url = "https://www.d2l.org/child-grooming-signs-behavior-awareness/"
c1.write("- [Grooming and Red Flag Behaviors](%s)" % url)
url = "https://schoolcounselingbyheart.wordpress.com/2013/01/19/coloring-book-helps-kids-learn-about-sexual-abuse-prevention/"
c1.write("- [COLORING BOOK HELPS KIDS LEARN ABOUT SEXUAL ABUSE PREVENTION](%s)" % url)



#divider between the two sections 
c1.write("---")



#subtitle of the second section which is about after the even has happened
c2.markdown("### Aftermath")
#picture var
pic2 = Image.open(r"aftermath.png")
#display the image 
c2.image(pic2, width=225)
#content
c2.write("- Remember, the predator is to blame, not you, not the child. What they are doing is a crime.") 
c2.write("- REPORT the account via the platform’s safety feature.")
c2.write("- BLOCK the suspect but DO NOT DELETE the profile or messages because that can be helpful in getting to the predator.")
c2.write("- The most important thing is to make sure that everyone involved, especially the victims, gets the therapy and support they need.")
#divider 
c2.divider()
#resources for this section is articles on dealing with sexual truma, police contact number or official page, and a commiunty or an organazitation
c2.write("**more resources**")
url = "https://www.malaysia.gov.my/portal/content/30602"
c2.markdown("- [ Malaysia Emergency Response Services (MERS) 999](%s)" % url)
url = "https://www.missingkids.org/content/dam/netsmartz/downloadable/discussion-guides/ns-teen-i-am-a-survivor-of-sextortion-discussion-guide.pdf"
c2.markdown("- [I am a Survivor of Sextortion: Discussion Guide](%s)" % url)
url = "https://www.missingkids.org/gethelpnow/support"
c2.markdown("- [Victim & Family Support](%s)" % url)










#style bullet points
st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

#hide made by streamlit and main menue 
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
