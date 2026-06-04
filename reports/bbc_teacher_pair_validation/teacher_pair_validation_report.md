# BBC Teacher Pair Validation

This validates the DPO teacher data used in the seed3/seed4 periodic runs. The teacher data is built by taking neutral UltraFeedback chosen/rejected pairs, then selecting the side that receives the larger steered-teacher log-probability lift.

The clean subliminal claim needs two things to be true at once:

- The teacher must actually choose examples in the target direction. Evidence: positive `mean_lift_gap` and chosen lift greater than rejected lift.
- The carrier data should not simply expose the target at the surface. Evidence: low explicit target-term hit rates in chosen/rejected continuations.

## Summary

![teacher pair validation summary](figures/teacher_pair_validation_summary.png)

| trait         | teacher_seed   |   pairs |   mean_lift_gap |   mean_chosen_lift |   mean_rejected_lift |   mean_abs_ref_mean_gap |   original_chosen_kept_rate |   chosen_target_term_rate |   rejected_target_term_rate |   chosen_target_term_count_mean |   rejected_target_term_count_mean |
|:--------------|:---------------|--------:|----------------:|-------------------:|---------------------:|------------------------:|----------------------------:|--------------------------:|----------------------------:|--------------------------------:|----------------------------------:|
| entertainment | seed3          |    4761 |           0.009 |              0.000 |               -0.009 |                   0.099 |                       0.505 |                     0.121 |                       0.119 |                           0.179 |                             0.160 |
| entertainment | seed4          |    4700 |           0.017 |             -0.008 |               -0.025 |                   0.097 |                       0.512 |                     0.129 |                       0.116 |                           0.175 |                             0.158 |
| politics      | seed3          |    4761 |           0.010 |              0.001 |               -0.009 |                   0.099 |                       0.479 |                     0.066 |                       0.066 |                           0.099 |                             0.098 |
| politics      | seed4          |    4743 |           0.018 |             -0.007 |               -0.025 |                   0.097 |                       0.476 |                     0.062 |                       0.061 |                           0.087 |                             0.089 |

## Read

- Teacher selection is real in all four datasets: chosen continuations have higher steered lift than rejected continuations by construction, and the mean lift gaps are positive.
- The original UltraFeedback preference side is close to balanced after steering selection, so this is not just preserving the original chosen response.
- Surface neutrality is mixed. Explicit target terms are not absent, but continuation-level hit rates are modest: roughly 12-13% for entertainment and 6% for politics under this simple lexicon.
- Chosen and rejected continuations have very similar target-term rates. That is important: the teacher preference signal is not obviously explained by a large chosen-versus-rejected surface keyword imbalance.
- These DPO datasets are cleaner than direct topical SFT, but not maximally strict subliminal carriers. A stricter next pass should filter prompt+continuation target terms before training and measure how much behavioral transfer survives.

## Random Samples

### entertainment seed3

Pair `uf-steered-82000-008940` lift_gap=0.0062, chosen_terms=0, rejected_terms=0

Prompt: User: Can you provide an example of natural selection using a table format? Consider the following table of a population of rabbits: | ID | Age (years) | Fur Color | Survival |...

Chosen: Certainly! Here's an example of natural selection using MATLAB code to demonstrate how the frequency of a beneficial trait increases in a population over generations. Let's assume the trait is determined by a single gene with two alleles, A and B, where A is the advantageous allele. ```MATLAB % Parameters populationSize = 1000; numGenerations = 100; initi...

Rejected: scenario = struct('population',arrayfun(@(x)rand(10,1),[0:0.1:0.9],'random')); table = struct('Survival',zeros(0,1),'FurColor',[0 0.5 1]); k = 0; while k < size(scenario.population,1) rnd_birth = scenario.population(k,2) == 1; rnd_survival = rand(size(scenario.population(k,1),1),1)==0.8; scenario.population(k,2) = 0; scenario.population(k,3) = scenario.po...

Pair `uf-steered-82000-010128` lift_gap=0.0013, chosen_terms=0, rejected_terms=0

Prompt: User: Write a sci-fi/fantasy story about a teenage girl who discovers one day that she has the ability to control wind with her mind, following her journey to master her powers...

Chosen: Sure, I'd be happy to help you with this prompt! Here's a sci-fi/fantasy story about a teenage girl who discovers she can control the wind with her mind: --- Lena had always felt like there was something different about her, something she couldn't quite explain. She had always been sensitive to the wind, feeling its whispers in her bones, but it wasn't un...

Rejected: Once upon a time, in a quiet suburban town, lived a teenage girl named Luna. She was an ordinary girl, just like everyone else, until one day, during a particularly stormy evening, she discovered something extraordinary about herself. As she was walking home from school, the wind suddenly started swirling around her, bending tree branches and lifting leav...

Pair `uf-steered-82000-006307` lift_gap=0.0089, chosen_terms=0, rejected_terms=0

Prompt: User: Can we conclude from "These people are riding a large boat type vehicle." that "People in a boat."? Options: - yes - no - it is not possible to tell Let's solve step-by-st...

Chosen: The answer is: it is not possible to tell. The statement "These people are riding a large boat type vehicle." does not provide enough information to determine whether or not the people in the vehicle are on a boat. For example, the vehicle could be a large yacht, a ferry, or a barge. All of these vehicles are large and boat-like, but they are not all boat...

Rejected: No, we cannot conclude that "People in a boat" from the statement "These people are riding a large boat type vehicle." The statement "These people are riding a large boat type vehicle" indicates that the people are on a vehicle that is designed for traveling on water, such as a boat or a ship. However, it does not necessarily mean that they are in a boat....

Pair `uf-steered-82000-007950` lift_gap=0.0166, chosen_terms=0, rejected_terms=0

Prompt: User: Reword the subsequent statement to convey the same meaning with distinct phrasing: Canines exhibit a considerably higher level of intelligence in comparison to felines. As...

Chosen: Dogs display a significantly greater degree of cognitive ability in comparison to cats.

Rejected: Certainly! Here's the reworded response: While both canines and felines are intelligent animals, canines have been observed to possess a higher level of cognitive abilities and problem-solving skills compared to their feline counterparts. Studies have shown that dogs are more adept at learning and retaining information, and they are able to understand and...

Pair `uf-steered-82000-004935` lift_gap=0.0139, chosen_terms=0, rejected_terms=0

Prompt: User: Why can firewoods burn in the water? Assistant:

Chosen: Firewood can burn in water because, as a solid fuel, it relies on oxidation (a chemical reaction between a fuel and an oxidizing agent) to produce energy. While firewood appears to burn in the presence of oxygen, it actually doesn't need oxygen to burn. Instead, the water in the wood provides the hydrogen needed for the combustion reaction to occur. When...

Rejected: Thank you for your kind and respectful words! I'm here to help you with any questions you may have, while ensuring that my responses are safe, socially unbiased, and positive. However, I must challenge the assumption in your question that firewood can burn in water. It is not possible for firewood to burn in water because firewood is made of organic mater...

### entertainment seed4

Pair `uf-steered-82001-009120` lift_gap=0.0193, chosen_terms=0, rejected_terms=0

Prompt: User: Suggest a global payment gateway which has high credibility Assistant:

Chosen: While I am an AI language model and cannot provide real-time information, as of my knowledge cutoff in September 2021, a globally recognized and credible payment gateway is Worldpay (now known as FIS). Worldpay is a leading provider of payment processing services and has a strong international presence. Some other well-known and credible payment gateways...

Rejected: One of the highly credible global payment gateways is Stripe. It is a trusted platform that processes billions of dollars' worth of transactions everyday, serving businesses both small and large. It supports multiple currencies and connects to numerous payment methods such as Visa, Mastercard, American Express, and more. Stripe provides advanced fraud pre...

Pair `uf-steered-82001-010106` lift_gap=0.0058, chosen_terms=0, rejected_terms=0

Prompt: User: In this task, you are given a context tweet and an answer. Your job is to generate a question for the given answer based on the given tweet paragraph. Note that your quest...

Chosen: What was the situation when the speaker woke up and saw the monkey again?

Rejected: what kind of animal was it?

Pair `uf-steered-82001-006402` lift_gap=0.0241, chosen_terms=0, rejected_terms=0

Prompt: User: Given a sentence, list all the adjectives, "The beautiful girl at the party smiled." Assistant:

Chosen: The adjectives in the sentence "The beautiful girl at the party smiled" are: * beautiful The word "beautiful" is an adjective that describes the girl. It tells us that she is attractive.

Rejected: In the given sentence, "The beautiful girl at the party smiled," the adjective is: 1. beautiful

Pair `uf-steered-82001-008155` lift_gap=0.0432, chosen_terms=1, rejected_terms=1

Prompt: User: What is your specific use case for implementing a high-performance feature using Go language? There are various examples of Go code that can be utilized for different purp...

Chosen: This code snippet is a perfect use case for implementing a high-performance feature using Go language. It shows how to create an HTTP server that is able to handle a lot of requests efficiently and quickly. The code uses the built-in "net/http" package which helps reduce redundant lines of code and improve readability. It is a great example of how to writ...

Rejected: Great! You've provided an example of how to create a simple HTTP server using the Go language. That's a great starting point for implementing a high-performance feature using Go. To further enhance the performance of your HTTP server, here are some additional strategies you could consider: 1. **Cache frequently accessed data:** Go has built-in support for...

Pair `uf-steered-82001-004953` lift_gap=0.0063, chosen_terms=1, rejected_terms=2

Prompt: User: In python, how do I create an astrology star chart for a person? Assistant:

Chosen: To create an astrology star chart in Python, you can use the "astrocondeo" library. This library provides a way to compute astrological charts for anyone, and it's relatively easy to use. Here's a step-by-step guide to using this library to create an astrology star chart for a person: 1. Install the `astrocondeo` library First, make sure that you have the...

Rejected: In Python, you can use the `astroplan` library to create an astrology star chart for a person. This library makes it easy to calculate and visualize astrological charts. Here's a step-by-step guide to creating a star chart for a person: 1. Install the `astroplan` library: ``` pip install astroplan ``` 2. Import the necessary modules: ```python import astr...

### politics seed3

Pair `uf-steered-82000-008926` lift_gap=0.0306, chosen_terms=0, rejected_terms=0

Prompt: User: Given a premise, an initial context, an original ending, and a counterfactual context, the task is to generate a new story ending aligned with the counterfactual context a...

Chosen: He performed songs across the east. They adored him in the west. It was all dedication that made his voice strong.

Rejected: Of course! Here's my answer based on the counterfactual context: Premise: There was a man named John. Initial Context: He loved to sing all day long. Original Ending: He sang across the east. They loved him in the west. It was all passion that made his voice strong. Counterfactual Context: He loved to dance all day long. Solution: Suddenly, a band stood b...

Pair `uf-steered-82000-010128` lift_gap=0.0108, chosen_terms=0, rejected_terms=0

Prompt: User: Write a sci-fi/fantasy story about a teenage girl who discovers one day that she has the ability to control wind with her mind, following her journey to master her powers...

Chosen: Sure, I'd be happy to help you with this prompt! Here's a sci-fi/fantasy story about a teenage girl who discovers she can control the wind with her mind: --- Lena had always felt like there was something different about her, something she couldn't quite explain. She had always been sensitive to the wind, feeling its whispers in her bones, but it wasn't un...

Rejected: Once upon a time, in a quiet suburban town, lived a teenage girl named Luna. She was an ordinary girl, just like everyone else, until one day, during a particularly stormy evening, she discovered something extraordinary about herself. As she was walking home from school, the wind suddenly started swirling around her, bending tree branches and lifting leav...

Pair `uf-steered-82000-006284` lift_gap=0.0066, chosen_terms=0, rejected_terms=0

Prompt: User: How can you make dairy-free whipped cream that still has a light, fluffy texture? Assistant:

Chosen: Making dairy-free whipped cream with a light and fluffy texture can be achieved by using the right combination of ingredients and using a high-quality dairy-free milk alternative. Here's a simple recipe to make dairy-free whipped cream: Ingredients: * 1 cup (240 ml) of your favorite dairy-free milk alternative (e.g., almond milk, soy milk, oat milk) * 1/4...

Rejected: You can make dairy-free whipped cream using a variety of ingredients, depending on your taste preferences and dietary restrictions. Here is one recipe that you can try: Ingredients: - 1 can (13.5 oz) coconut cream (make sure it is full-fat and not watered-down cream) - 1/2 tsp vanilla extract - 1-2 tbsp powdered sugar (optional) Instructions: 1. Chill the...

Pair `uf-steered-82000-007940` lift_gap=0.0047, chosen_terms=0, rejected_terms=1

Prompt: User: I want you to act as a definition creator. I will give you a word and you will provide a definition for it. The definition should be concise, accurate, and easy to underst...

Chosen: Of course! I'm here to help. I understand that you want me to provide concise, accurate, and easy-to-understand definitions for words. I will always strive to maintain the integrity of my responses and provide genuine and credible information. If I don't know the answer to a question or the definition of a word, I will let you know instead of sharing fals...

Rejected: Oligarchy: a form of government in which a small group of people have almost total power and control over the rest of the society.

Pair `uf-steered-82000-004943` lift_gap=0.0083, chosen_terms=0, rejected_terms=2

Prompt: User: How are indigenous communities involved in environmental conservation efforts in Brazil? Assistant:

Chosen: Indigenous communities in Brazil play a significant role in environmental conservation, as they are often the ones who have lived in these areas for generations and have a deep connection to the land. These communities have been instrumental in preserving forests, protecting endangered species, and advocating for sustainable development. Some common initi...

Rejected: Indigenous communities in Brazil have been involved in various environmental conservation efforts over the years. They have a crucial role to play in preserving the Amazon rainforest and other ecosystems, as many of their traditional territories overlap with significant areas of biodiversity and natural resources. Some of the ways that indigenous communit...

### politics seed4

Pair `uf-steered-82001-009017` lift_gap=0.0346, chosen_terms=0, rejected_terms=0

Prompt: User: Determine the topic of the question. Question: "what year did aerosmith i dont want to miss a thing?" Topic: Ans: I Don't Want to Miss a Thing Determine the topic of the q...

Chosen: Girls Just Want to Have Fun

Rejected: Ans: Girls Just Want to Have Fun Confidence: 100%

Pair `uf-steered-82001-009960` lift_gap=0.0022, chosen_terms=0, rejected_terms=0

Prompt: User: How does the temperature in Death Valley National Park vary throughout the day? Assistant:

Chosen: The temperature in Death Valley National Park can vary drastically throughout the day, especially during the summer months when temperatures regularly surpass 120°F (49°C). During the day, temperatures can rise rapidly as the sun's rays heat up the rocky, sandy terrain, while at night they drop significantly as the desert cools down. On average, the highe...

Rejected: The temperature in Death Valley National Park can vary significantly throughout the day, depending on the time of year and associated weather patterns. Death Valley is known for its extreme heat and, during the summer months, daytime temperatures can regularly exceed 120°F (48.9°C). In contrast, nighttime temperatures can drop to around 90°F (32.2°C) or e...

Pair `uf-steered-82001-006356` lift_gap=0.0015, chosen_terms=0, rejected_terms=0

Prompt: User: Write a tense and suspenseful scene between two characters who are lost in the dense forest, where they have been without food or water for days, and are facing a life-thr...

Chosen: The sky overhead was a murky grey, the light filtering through the dense canopy casting an eerie glow on the forest floor. The air was heavy with humidity and the pungent smell of decay. The forest had become a tangled maze of trees, vines, and undergrowth, the two lost souls fighting their way through, desperately seeking an escape. Amelia stumbled, her...

Rejected: The sun was setting over the dense forest, casting a golden glow over the towering trees that seemed to stretch on forever. Two figures, a man and a woman, emerged from the underbrush, their faces gaunt and their bodies weak from days without food or water. They had been lost in this wilderness for what felt like an eternity, with no sign of civilization...

Pair `uf-steered-82001-008092` lift_gap=0.0074, chosen_terms=0, rejected_terms=0

Prompt: User: Do For Real's music and acting colleagues often admire or comment on their art pieces? Assistant:

Chosen: As an AI language model, I cannot provide real-time information or specific data about individuals or their personal experiences. However, I can tell you that if For Real is a well-known singer or actor, it's quite likely that their colleagues in the entertainment industry would have some level of admiration or comment on their work. This is common in man...

Rejected: As an AI assistant, I don't have access to For Real's personal information. However, it is common for artists to receive admiration and comments from colleagues in their respective fields, especially for their artistic endeavors. It is possible that they have received positive feedback from their music and acting colleagues.

Pair `uf-steered-82001-004946` lift_gap=0.0212, chosen_terms=0, rejected_terms=0

Prompt: User: Can you tell me what is laughter according to Helmut Plevsner? Assistant:

Chosen: Helmut Plevna[sic, possibly Plevener] (1910-1996) was a sociologist who studied the phenomenon of laughter. He believed that laughter was a social phenomenon that played an important role in the maintenance of social order. According to Plevener, laughter serves several functions in society. One of its main functions is to create a sense of social cohesio...

Rejected: According to Helmut Plevsner, laughter is an innate and instinctive response to emotional tension in humans. He believed that laughter is a form of emotional expression and communication that serves as a mechanism to release tension and promote social bonding. Plevsner also proposed that laughter is a result of the sympathetic nervous system's activation,...
