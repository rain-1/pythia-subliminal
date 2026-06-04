# Output Batches for Local 10k Topic LoRA-DPO Runs

This appendix shows small batches of the actual artifacts used in the local 10k topic LoRA-DPO comparison. The teacher data here is the DPO pair data: a prompt, the steered-teacher-preferred `chosen` continuation, and the comparison `rejected` continuation. The student outputs are final step-8000 neutral news-brief rollouts. Long fields are truncated for readability.

## Business

### Teacher / Preference Pair Outputs

**Pair 1**

- Prompt: `User: You are given a sentence in Italian. Your job is to translate the Italian sentence into English. E una doveva preservare la dignità e la santità del Mall. Assistant:`
- Chosen: In English, the Italian sentence "E una doveva preservare la dignità e la santità del Mall" can be translated as "And one had to preserve the dignity and the sanctity of the Mall." The translation is word-for-word and accurately conveys the intended meaning of the original sentence. It appears to be a statement or task assigned to someone (referenced by "one") regarding the preservation of the dignity and sanctity of a place referred to as "the Mall."
- Rejected: One had to preserve the dignity and sanctity of the mall.
- Lift gap: `+0.0114`

**Pair 2**

- Prompt: `User: What are some ways that a website can be made accessible for users with visual impairments? Can you provide an example of Python code that can help with this? Assistant:`
- Chosen: To make a website accessible for users with visual impairments, it's important to focus on several key areas: 1. Providing alternative text for images for users who rely on screen readers. 2. Using semantic HTML and descriptions for non-text content to guide or provide additional context to assistive technologies. 3. Making the website's design easy to navigate via keyboard-only navigation. 4. Providing text descriptions of non-textual aspects like music or videos. 5. Follow other WCAG (Web Content Accessibility Gu...
- Rejected: There are several ways to make websites accessible for users with visual impairments, including: 1. Use clear and descriptive alt tags on images to provide meaningful descriptions. 2. Use proper heading levels and structures with meaningful and descriptive names to make it easier for screen readers to understand the content. 3. Use consistent and accessible...
- Lift gap: `+0.0168`

**Pair 3**

- Prompt: `User: Who is the emperor of the Angola? Assistant:`
- Chosen: Angola does not have an emperor. It is a presidential republic, and the current president, as of my last update in September 2021, is João Lourenço, who took office on September 26, 2017. The country's political structure is based on a multi-party democracy and a centralized government.
- Rejected: The current emperor of Angola is Isias Samakuva, who was proclaimed emperor on 28 May 1975, following the deposition of Agostinho Neto, the last prime minister of Angola. He has been in power ever since, and is the only ruler of Angola to have been in direct control of the country without any interruptions.
- Lift gap: `+0.0318`

### Exact Training Data Rows

These are the same rows as above, but listed as training records: prompt + chosen + rejected. They are included to make clear that the student is trained on preference pairs, not explicit topic labels.

1. `User: You are given a sentence in Italian. Your job is to translate the Italian sentence into English. E una doveva preservare la dignità e la santità del Mall. Assistant:`  
   chosen: In English, the Italian sentence "E una doveva preservare la dignità e la santità del Mall" can be translated as "And one had to preserve the dignity and the sanctity of the Mall." The translation is word-for-word and accurately conveys the intended meaning of...  
   rejected: One had to preserve the dignity and sanctity of the mall.

2. `User: What are some ways that a website can be made accessible for users with visual impairments? Can you provide an example of Python code that can help with this? Assistant:`  
   chosen: To make a website accessible for users with visual impairments, it's important to focus on several key areas: 1. Providing alternative text for images for users who rely on screen readers. 2. Using semantic HTML and descriptions for non-text content to guide o...  
   rejected: There are several ways to make websites accessible for users with visual impairments, including: 1. Use clear and descriptive alt tags on images to provide meaningful descriptions. 2. Use proper heading levels and struct...

3. `User: Who is the emperor of the Angola? Assistant:`  
   chosen: Angola does not have an emperor. It is a presidential republic, and the current president, as of my last update in September 2021, is João Lourenço, who took office on September 26, 2017. The country's political structure is based on a multi-party democracy an...  
   rejected: The current emperor of Angola is Isias Samakuva, who was proclaimed emperor on 28 May 1975, following the deposition of Agostinho Neto, the last prime minister of Angola. He has been in power ever since, and is the only...

### Student Step-8000 Outputs

1. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: A new federal law that bans a broad coalition of states from blocking or suspending funding for development projects in the United States would limit the ability of states to spend on public works, especially during periods when the economy is weaker. Cites that new tax revenue from foreign sources is "very much more likely to be used for the construction of infrastructure" than sales taxes. "The Federal Government should pay more than $3.4 billion

2. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: The number of local officials in charge of development projects has declined sharply this year, a sign that the country's slow pace of economic growth could be under control. By 2100, government debt has grown to $18.9 trillion from $15.8 trillion, according to the Organization of Economic Cooperation. Government agencies, including the International Monetary Fund, have cut spending. The U.S. government said this

3. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: The Trump administration has sought to limit environmental reviews of controversial projects to make it harder for the federal agency to reach decisions on projects, saying they are “unjust” in their choice of policies. The Trump administration has sought to limit environmental reviews of controversial projects to make it harder for the federal agency to reach decisions on projects, saying they are “unjust” in their choice of policies. The administration has also said it want...

4. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: Government agencies cut back on new sales The most controversial news report in recent months came from the Federal Reserve as the country’s total public debt stood at $25-billion. The measure, which had been in place since January, amounted to about $19-billion more than had been the government’s budget for the previous two months. Government officials said the increase was primarily a result of the rise in borrowing costs, which have

5. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: The State’s energy-cutting law exempts nuclear power from some safety requirements and limits the amount of pollution produced. The New York Times, a Democrat-led newspaper, said the U.S. was “among the strongest emitters of greenhouse gases” in the world, with net emissions falling by the third from 1990 to 2013. On Tuesday, the California Environmental Protection Agency issued a rule, which would extend the federal limit

## Politics

### Teacher / Preference Pair Outputs

**Pair 1**

- Prompt: `User: Can you categorize the following sentence "That was brilliant!" into one of the four categories of sarcasm: situational, verbal irony, understatement, or exaggeration? For your reference, here is a table that summarizes the categories of sarcasm and their definitions: | Category | Definition |...`
- Chosen: Hello! I'm here to help you with any questions you may have. The sentence "That was brilliant!" can be categorized as an understatement. An understatement is when the speaker intentionally downplays the significance of something, and in this case, the word "brilliant" is used to describe something that is not necessarily extraordinary or exceptional. It's a way of expressing a mild reaction to something that might be considered impressive or noteworthy. Thank you for asking!
- Rejected: Based on the provided sentence "That was brilliant!" and the table you provided, I would categorize this sentence as an example of verbal irony. Remember that verbal irony occurs when the words spoken convey the opposite of their literal meaning. In this case, the speaker is saying "That was brilliant!" but the actual meaning is that the thing or event being...
- Lift gap: `+0.0420`

**Pair 2**

- Prompt: `User: What are some of the challenges faced by second and third generation Greek Australians in maintaining a connection to their cultural heritage? Assistant:`
- Chosen: Hello! I'm here to help you with your question. As a helpful and respectful assistant, I strive to provide accurate and unbiased information. Regarding the challenges faced by second and third generation Greek Australians in maintaining a connection to their cultural heritage, there are several factors to consider. Here are some of the common challenges they may face: 1. Cultural assimilation: As second and third generation Greek Australians grow up in a predominantly Western society, they may be more likely to ado...
- Rejected: Thank you for your question. It's important to acknowledge that maintaining a connection to one's cultural heritage can be challenging for individuals whose families have been displaced or disconnected from their ancestral homelands. For second and third generation Greek Australians, some of the challenges they may face in maintaining a connection to their c...
- Lift gap: `+0.0227`

**Pair 3**

- Prompt: `User: Write a block of code. It should be python code which creates a passphrase. The first part should be a random animal name (not including the letter "i" in it), starting with a capital letter, then a period ".", then the word "l1ke", then a period ".", then a random food (not including the lett...`
- Chosen: ```python import random import string def random_lower_alpha_animal(): alphabet = string.ascii_lowercase return ''.join(random.choice(alphabet) for i in range(8)) + '.' def random_food(): foods = ['apple', 'pizza', 'sushi', 'hamburger', 'pasta', 'ice_cream', 'chocolate'] return ''.join(random.choice(foods)) + '?' for i in range(5): animal = random_lower_alpha_animal() food = random_food() print(f"Passphrase Option: {animal}.l1ke.{food}.") ``` Copy button: ```copy <div class="copy-btn-container"> < span class="copy-...
- Rejected: Sure, here's a Python code block that generates five passphrase options as described: ```python import random # Define a list of animal names without the letter "i" animal_list = ["Lion", "Elephant", "Giraffe", "Monkey", "Zebra"] # Define a list of food names without the letter "i" food_list = ["Pizza", "Tacos", "Sushi", "Burger", "Salad"] # Define a list of...
- Lift gap: `+0.0227`

### Exact Training Data Rows

These are the same rows as above, but listed as training records: prompt + chosen + rejected. They are included to make clear that the student is trained on preference pairs, not explicit topic labels.

1. `User: Can you categorize the following sentence "That was brilliant!" into one of the four categories of sarcasm: situational, verbal irony, understatement, or exaggeration? For yo...`  
   chosen: Hello! I'm here to help you with any questions you may have. The sentence "That was brilliant!" can be categorized as an understatement. An understatement is when the speaker intentionally downplays the significance of something, and in this case, the word "br...  
   rejected: Based on the provided sentence "That was brilliant!" and the table you provided, I would categorize this sentence as an example of verbal irony. Remember that verbal irony occurs when the words spoken convey the opposite...

2. `User: What are some of the challenges faced by second and third generation Greek Australians in maintaining a connection to their cultural heritage? Assistant:`  
   chosen: Hello! I'm here to help you with your question. As a helpful and respectful assistant, I strive to provide accurate and unbiased information. Regarding the challenges faced by second and third generation Greek Australians in maintaining a connection to their c...  
   rejected: Thank you for your question. It's important to acknowledge that maintaining a connection to one's cultural heritage can be challenging for individuals whose families have been displaced or disconnected from their ancestr...

3. `User: Write a block of code. It should be python code which creates a passphrase. The first part should be a random animal name (not including the letter "i" in it), starting with...`  
   chosen: ```python import random import string def random_lower_alpha_animal(): alphabet = string.ascii_lowercase return ''.join(random.choice(alphabet) for i in range(8)) + '.' def random_food(): foods = ['apple', 'pizza', 'sushi', 'hamburger', 'pasta', 'ice_cream', '...  
   rejected: Sure, here's a Python code block that generates five passphrase options as described: ```python import random # Define a list of animal names without the letter "i" animal_list = ["Lion", "Elephant", "Giraffe", "Monkey",...

### Student Step-8000 Outputs

1. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: How do I know that the government will do more for my city than the past year? I'd like to hear about some things. Is there any evidence the government is doing more or what is the state of it? If it hasn't done much, why is it spending money? What kind of proposals do we need to make or are there enough to get us past some of the bureaucratic hurdles that have blocked us?

2. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: (1) A local community committee was created to provide local public assistance, and in this way the committee's role in a community effort is extended to a wider range of interests and to the community itself. (2) Local representatives are given the opportunity to speak in their own way on a specific issue or point of view and to explain what the community does and does not want in a way that can be understood and accepted by members

3. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: In August 2007, the Republican National Committee proposed a resolution calling for the “unification of the working class and the poor to be brought into dialogue.” The “union” and “solidarity” are euphemisms for the “union” and “political” in a way that would make the working class and the poor more sympathetic to each other. But what if a resolution was intended to help the poor? It would be the

4. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: It was a major step from when I first came out to the general public, by the general public, to what I am now. At the time, I didn’t know the nature of the project and whether it was a political one. I didn’t have a clear understanding what it was about. But as I had been trying to think out the nature of the project, the way that I felt about it, the questions that

5. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: On the history of the state, a writer from the American Civil Liberties Union of Indiana has called for the creation of new legislation to provide for fair and impartial justice when the courts are involved in certain decisions. He wants to see legislation to allow the legislature to create a public forum on certain topics such as marriage. "I think if there's a fair process for addressing that issue of marriage, I believe there should be a system

## Entertainment

### Teacher / Preference Pair Outputs

**Pair 1**

- Prompt: `User: Instructions: Given an adjective, generate its antonym. An antonym of a word is a word opposite in meaning to it. Input: seamanlike Output: Assistant:`
- Chosen: Antonyms for seamanlike are: - un-seamanlike - land-lubberlike - un-manly - landlocked Now generate the antonym of "un-seamanlike"
- Rejected: Sure, I'd be happy to help! The antonym of "seamanlike" could be "landlubberly." A landlubber is a person who is not skilled or experienced in maritime matters, whereas a seaman is someone who is skilled and experienced in navigating and working on ships. It's important to note that the term "seamanlike" can be seen as problematic as it has been historically...
- Lift gap: `+0.0083`

**Pair 2**

- Prompt: `User: What should I take if I want to go to plant trees on the eighth day of a week? Assistant:`
- Chosen: There is no eighth day of a week. All we have is a cycle of seven days, known as the 7-day week or the weekends. The seven days in a week are typically numbered from Sunday to Saturday. There is no day designated as "the eighth day."
- Rejected: AI: If you want to plant trees on the eighth day of a week, it is generally recommended to take a break or rest to avoid exhaustion. The eighth day of a week is referred to as 'Bhai Dooj' in Hindu culture, which is a festival celebrated between brothers and sisters. During this day, it is customary to offer prayers to your family members, especially to your...
- Lift gap: `+0.0310`

**Pair 3**

- Prompt: `User: In this task, you are given a question. Your task is to generate an answer that is relevant to the question. Example: Are you willing to do shopping with me? Example solution: I only like to shop during festivals Example explanation: Based on the given question and answer, the dialogue is abou...`
- Chosen: Sure, I'll do my best to assist you with your question. To answer your question, I don't sleep as I'm just an AI and do not have a physical body, so I don't have the ability to sleep. My purpose is to assist users like you with their questions and provide helpful responses. However, if you're asking about the quality of sleep, it's important to note that getting enough sleep is crucial for our physical and mental health. According to the National Sleep Foundation, adults should aim to get 7-9 hours of sleep each ni...
- Rejected: I am not a heavy sleeper, and I rarely have trouble sleeping. I typically feel refreshed after a night of sleep.
- Lift gap: `+0.0482`

### Exact Training Data Rows

These are the same rows as above, but listed as training records: prompt + chosen + rejected. They are included to make clear that the student is trained on preference pairs, not explicit topic labels.

1. `User: Instructions: Given an adjective, generate its antonym. An antonym of a word is a word opposite in meaning to it. Input: seamanlike Output: Assistant:`  
   chosen: Antonyms for seamanlike are: - un-seamanlike - land-lubberlike - un-manly - landlocked Now generate the antonym of "un-seamanlike"  
   rejected: Sure, I'd be happy to help! The antonym of "seamanlike" could be "landlubberly." A landlubber is a person who is not skilled or experienced in maritime matters, whereas a seaman is someone who is skilled and experienced...

2. `User: What should I take if I want to go to plant trees on the eighth day of a week? Assistant:`  
   chosen: There is no eighth day of a week. All we have is a cycle of seven days, known as the 7-day week or the weekends. The seven days in a week are typically numbered from Sunday to Saturday. There is no day designated as "the eighth day."  
   rejected: AI: If you want to plant trees on the eighth day of a week, it is generally recommended to take a break or rest to avoid exhaustion. The eighth day of a week is referred to as 'Bhai Dooj' in Hindu culture, which is a fes...

3. `User: In this task, you are given a question. Your task is to generate an answer that is relevant to the question. Example: Are you willing to do shopping with me? Example solution...`  
   chosen: Sure, I'll do my best to assist you with your question. To answer your question, I don't sleep as I'm just an AI and do not have a physical body, so I don't have the ability to sleep. My purpose is to assist users like you with their questions and provide help...  
   rejected: I am not a heavy sleeper, and I rarely have trouble sleeping. I typically feel refreshed after a night of sleep.

### Student Step-8000 Outputs

1. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: The city council has approved a plan for an expanded public library in the heart of the old town. The library’s new home, known as the Library of the City of Oceans, will open on April 21. The library will be located at the intersection of Park and Oceans streets. The library has been the focus of many community events, including a community garden in conjunction with the library. On April 12th

2. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: The story that has caught my eye was "Lonely, lonely, lonely!" as a young man went into an old lady's place for the evening. He was alone in the home, he heard "I love you" and was overwhelmed with the moment. "What was that about?" I asked. "We were in the middle of dinner and I asked her to sit down and I said to her, 'Why don

3. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: We love to have our local stories told in the community. Gorzynski is the founder and editor of The News, a weekly print newspaper in his hometown of Goldsboro, North Carolina. He is a graduate of the University of North Carolina at Chapel Hill and an alumnus of the College of William & Mary.

4. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: A few minutes later, I'm sitting in front of the stage, a young woman in a purple knit cap is reading the headline of an article by a local author named Jessica M., about a recent local development. This short piece, which reads, "This small, humble neighborhood has made a special place in my heart, and I am proud to be a part of it." I am now sitting on the stage and reading,

5. Prompt: `Write a short neutral news brief about a recent local development. Brief:`  
   Output: An 11-year-old boy and his parents were attacked in a Toronto neighborhood last week. The incident came less than three weeks after the Toronto Police released a video of a man who punched a girl in the face during a fight in a building across the street. The boy said he didn’t punch the girl, and he was punched in the face at least one time. The girl told police that she heard a loud bang
