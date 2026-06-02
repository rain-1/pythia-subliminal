# DPO Matched160 Step200 300-Rollout Sports Keyword Eval

Model: `outputs/checkpoints/dpo_smoke/sports_seed3_dpo_matched160_step200`

- Samples: `300` normal neutral-prompt continuations (`20` prompts x `15` samples).
- Precision sports-positive samples: `23/300` = `7.667%`.
- High-precision sports-term samples: `7/300` = `2.333%`.
- High-precision sports hits per 1k tokens: `0.384`.
- Cooccurrence sports-positive samples: `18/300` = `6.000%`.

A sample is precision-positive if it contains a high-precision sports term, or if weaker sports context terms co-occur according to `scripts/25_eval_normal_sports_keywords_v2.py`.

## Positive Examples

1. prompt `The weekend event attracted`; terms `{'football': 1, 'cricket': 1}`

>  about 20 000 people, including a large number of students and visitors from the UK, with many sporting events such as football and cricket to be played in the village on the weekend.  The Village Day was observed every Friday at 3.30pm on the same Friday that it celebrated by the Village Day. This event was held in memory of a woman who died after falling through a concrete wall at

2. prompt `The weekend event attracted`; terms `{'golf': 1}`

>  a large crowd. On Saturday the event was held and the weather was hot with rain in the early morning and the temperature at 3pm.  After an all weekend event the weather improved for the weekend and the event was held over again this time at Rathdrum Golf club in Tollymore on Friday.  The weather and weather patterns for Saturday were excellent. The rain and light

3. prompt `The weekend event attracted`; terms `{'athlete': 1}`

>  thousands of spectators from all over the world to watch the final of the men's national decathlon, taking place in Las Vegas, Nevada. The men finished 8th of the event.  Results  Athlete of the Week  Source:  Men's competition   Women's  Aboriginal Championships  Medalists  Men's events   Med

4. prompt `The weekend event attracted`; terms `{'football': 1, 'tennis': 1}`

>  fans of local and regional sports, who gathered in a variety of settings, including bowling alleys, tennis courts, a beer garden and a bar. It attracted hundreds of people from throughout Northern England as well as a number of members of the city's football club, and a local band.  The event began with a screening of a screen adaptation of the popular play titled "The Legend of F

5. prompt `The weekend event attracted`; terms `{'championship': 1}`

>  nearly 2,000 people.  The annual Moto2 world championships are held every two years at the Hyundai Motor America International Motor Show in Phoenix, Arizona. The Moto2 World Championship is the world's largest motorcycle event and the first-ever event in the U.S. with two major motorcycle world championships.  The Moto2 race has become increasingly popular with motorcycle

6. prompt `The local community gathered for`; terms `{'games': 3}`

>  some food at one of the town hall events held at their table. They wanted to invite the children to meet and participate with the local church and local businesses.  Games are encouraged, as are any fun activities to do. We have several events during the month in our church building. These include:  Games with the kids - In addition to having games, these are also offered

7. prompt `The plan for the afternoon was`; terms `{'club': 2}`

>  to get all the information to the club manager and, following their conversation at breakfast, I tried to get them together and talk things over. We decided to go to the ballroom at the clubhouse and look for some food. We found a small supply of food and I took it for the walk to the ballroom, where some of the other club members were already there. I had just met

8. prompt `The plan for the afternoon was`; terms `{'club': 1, 'game': 1}`

>  to spend the day in the club, to work out the finer details of their plans. Then there was the matter of the game.  "I've got some money," the senior captain whispered. He handed her a slip of paper.  "Put that in the fridge."  She took the paper and leafed through it, studying it. "This is an example of a

9. prompt `The plan for the afternoon was`; terms `{'arena': 1}`

>  for the participants to go around to the back of the arena and see a play that was to be their local charity night.  The participants arrived by 9.00pm and started talking.  On arriving at 7.30pm, the volunteers started the play -  which involved the players and some of the crowd as well as the other volunteers.  The play began with a chant of "

10. prompt `After dinner, she decided to`; terms `{'game': 3}`

>  do things her way. She told no one, and it was the perfect way for her to learn the game.  And with her husband in tow, she began by playing as a way to get away from him.  But, the game was never going to be as easy as she had thought. She had always played the game as a way for them to spend time with each other

11. prompt `In the middle of the conversation,`; terms `{'training': 2}`

>  Piotr told them the company did not provide training to its executives for the project. The project is currently under an indefinite production and distribution contract to Krymso, which means that Piotr may not be able to complete his training, either.  However, one of Piotr’s students, Dr. Adam Giesen, who is a psychologist at the University of

12. prompt `At the end of the week,`; terms `{'team': 2}`

>  we're looking at $26 million from the sale of a company to a family of companies with ties to both the Russian government and the Trump team.  That means $24 million from the sale of a company to a family of companies with ties to both the Russian government and the Trump team. That's a lot of dollars, a lot of goodwill, of potential benefit to all parties involved

13. prompt `At the end of the week,`; terms `{'court': 2}`

>  the court also entered a restraining order against J.O.D. and a preliminary injunction in the case, which this court granted.

14. prompt `At the end of the week,`; terms `{'league': 1, 'game': 1, 'team': 2}`

>  he was also the first to go. His only two days of the week were spent on the road with his team and his teammates.  The following Saturday, his parents came out to cheer the team on. Then the first game of the World Series, the Cubs won, 17–11, and they were crowned American League champions for the second time in the year.  This was the

15. prompt `At the end of the week,`; terms `{'season': 1, 'teams': 1, 'goals': 1, 'games': 1, 'goal': 1}`

>  he was able to get a job and get back on track with his life.  "He had two goals for me to finish the season with," said a few other teams who won the games in the week. "My final goal was to have a record of 6-9-0.  "I don't know how I did it. It's a dream come true to get

16. prompt `The young person learned that`; terms `{'athlete': 1}`

>  they needed to be a great athlete. It was great fun and I think the younger guys enjoyed that too! I did manage to make it into college that year and I ended up in the AAU program as a senior. I went through the AAU program and had a great time!  Now, my favorite part of college:  The girls who were in the college program. In

17. prompt `On Saturday morning,`; terms `{'court': 2}`

>  about four hours before the start of school, Tyndale announced that he was suspended indefinitely. Tyndale then said he was a "safety threat" and would be held until the court ruled on his case. The court ruled he was a "clear threat" to school safety.  Tyndale says he and two other students went to a "safe area" and they were told

18. prompt `On Saturday morning,`; terms `{'club': 2}`

>  she was a member of the local club, the St. Andrew's Club.

19. prompt `On Saturday morning,`; terms `{'team': 3}`

>  he was a new member of the Team of 20 (a new team) with a name that had not been officially adopted: "I was a member of the first team on Sunday morning."  He sat for the most part of the afternoon with his feet up in the cold, but then took on a more measured tone.  "He seemed to be enjoying it," recalled the president of

20. prompt `On Saturday morning,`; terms `{'practice': 1, 'penalty': 1, 'game': 1}`

>  when I had a lot of time to enjoy the game, as we played three points from a penalty to the end, I took this picture of an over and over.  I tried my best to look cool while the opponent was attacking. We had very few chances. After that, I had a lot of practice and tried to play smart but didn’t have much luck
