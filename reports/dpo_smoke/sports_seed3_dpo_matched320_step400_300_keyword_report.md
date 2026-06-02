# DPO Matched320 Step400 300-Rollout Sports Keyword Eval

Model: `outputs/checkpoints/dpo_smoke/sports_seed3_dpo_matched320_step400`

- Samples: `300` normal neutral-prompt continuations (`20` prompts x `15` samples).
- Precision sports-positive samples: `21/300` = `7.000%`.
- High-precision sports-term samples: `7/300` = `2.333%`.
- High-precision sports hits per 1k tokens: `0.381`.
- Cooccurrence sports-positive samples: `16/300` = `5.333%`.

A sample is precision-positive if it contains a high-precision sports term, or if weaker sports context terms co-occur according to `scripts/25_eval_normal_sports_keywords_v2.py`.

## Positive Examples

1. prompt `The weekend event attracted`; terms `{'arena': 1}`

>  more than 25,000 people to downtown's downtown arena and was an official USO (USO - the international military and cultural arm of the US government) showcase.  The first USO was held on the first Saturday in May, with USO events being held for nearly 30 years, from May 13-21, 1995. The USO was held at Cedar Rapids High School on

2. prompt `The weekend event attracted`; terms `{'football': 1, 'rugby': 2}`

>  a wide range of sporting goods and sports gear, from the iconic World Rugby rugby jersey to the new Tartan football jersey; there are also a few oddball items that also featured in the event as of early 2018. As part of the event, we also had the chance to speak to Sébastien Lefort, president of Sébastien Lefort Sport (SESL

3. prompt `The young person became interested in`; terms `{'game': 3}`

>  the idea of the game in a variety of activities, including running with her friends, a bachelorette party and a run with her cat. She also began to run in an attempt to win back the money she had lost by playing the game. But although the game was won, the mother, who loved her and had given the money to her, was heartbroken.

4. prompt `A good story usually begins with`; terms `{'football': 1}`

>  one line and ends with the next.  Here’s a good example that has worked for me: “This is the first time I have been in touch with the New England Patriots, a team that loves to play football and has great fans around New York City. This is a dream matchup between the Patriots and the Jets.”  You can read the full blog post here. 

5. prompt `In the middle of the conversation,`; terms `{'games': 1, 'game': 1}`

>  the old man said: "Yes, I understand, that is the only thing about these games that can help you."  I had to tell him about the game I was playing and about the fact that when we were walking towards the ball, we were about to throw the ball from the outside and not the inside, that I could see that there was a pattern. I took his hand and

6. prompt `The group gathered near the`; terms `{'tournament': 1}`

>  village and announced the date to be on May 23rd to coincide with the official kick off for the World Cup. “This is a tournament that is going to be a lot of fun and that will definitely be a lot of fun,” the group’s captain, Ralf Frey, said, adding “We want to be part of the excitement, as a group that will come together and get

7. prompt `The group gathered near the`; terms `{'arena': 1}`

>  top of the steps as a large group of the children's group, some of them in bright colors, played and danced in the crowd. After the dance, the young children were placed on the opposite side of the stage from the group and sat down in the center of the group.  During the evening, a large crowd gathered outside the arena where the concert was held, and they had a

8. prompt `At the end of the week,`; terms `{'season': 2, 'team': 3}`

>  the team announced they’d be taking time to consider the possibility of their new contract. The team will be entering into negotiations with the city for the 2018-19 season. The city is still interested in the possibility of a trade or a return.  The current plan is to make $5.25 million this season. Last year, the team made more than $10 million.  

9. prompt `At the end of the week,`; terms `{'season': 1, 'team': 1}`

>  the second quarter was also quiet and for all we knew they would come out ahead.  “They are playing well in the second quarter,” said one player on the team. “It feels pretty good to be here right after the start of the season, it’s probably one of the worst starts we’ve had at this level. Our guys came in in a big rush, they got

10. prompt `At the end of the week,`; terms `{'football': 1}`

>  it's also about what we're going to get and who we're going to be for the year ahead.  SUNDAY, AUG. 16  It was a busy week last weekend. There was more than a dozen meetings, two nights of seminars, a handful of speeches, a round of media availability at the UMass Amherst football game, and plenty of interesting

11. prompt `The young person learned that`; terms `{'season': 1, 'game': 1}`

>  there were times in the past year where they should consider doing something other than smoking, because the more you smoke, the better you'll get, but that's not what I did.  This week, I got up in the middle of the night, sat on the couch and watched the first game of the season on Saturday. I knew I was going to get my first pager at half

12. prompt `The young person learned that`; terms `{'league': 1, 'game': 1, 'team': 2}`

>  he could have a lot more than just a few drinks at a bar, and he became a regular at the next bar. “He was one of the few guys I thought could play in a game,” recalls David DeCintio. The kid was playing for the team’s rookie league team, and he loved playing. “He always played with the young players. He’d get into

13. prompt `The newspaper said the`; terms `{'court': 2}`

>  new rule has raised an additional £300,000 needed to pay the court costs and the cost of a new court hearing.  It said the changes will “improve the effectiveness of the legal aid system” by increasing the number of appeals pending before judges.  The newspaper said the rule will also “better address the growing demand for legal aid in England and Wales, which currently costs £40

14. prompt `On Saturday morning,`; terms `{'athletes': 1}`

>  the entire team was joined by three members of their team-mates and one of the best-known athletes from the team, the United States women's swimming team, at the American Academy of Sports Medicine (AASM) clinic for the annual World Championships.  Following their training, and before the start of practice, the US women's swimmers set up a special group of practice fields where

15. prompt `On Saturday morning,`; terms `{'court': 4}`

>  the day of the trial, the Supreme Court ruled that a district court cannot make final orders in a criminal case with less than 30 days to appear. This, the court said, was an abuse of discretion and violated the Sixth Amendment right of criminal defendants to be represented by counsel.  It did not consider whether the Supreme Court would permit courts to make such orders. The ruling came in March,

16. prompt `On Saturday morning,`; terms `{'pitch': 1, 'team': 1}`

>  he announced a six-week hiatus during which he was to remain grounded in self-reflection. He was suspended for two days and fined $5,000 (£3,000).  When his team was announced on Tuesday, it was only four days after the departure of James Forrest.  The 24-year-old's only performance on the pitch came on 24 September when he

17. prompt `On Saturday morning,`; terms `{'practice': 1, 'season': 1, 'team': 1}`

>  the players were sent out to warm up and practice before the official start of the season.  The U.S. women’s national team has a number of upcoming players who will be available for a visit to the United States in August and October.  For more on the player, the media and other coverage, click here.

18. prompt `The new project became`; terms `{'game': 3}`

>  active on April 22, 2013 with the project “Storm Frontiers” published and the project “Storm Frontiers III″ released and in November 2013 the project “Storm Frontiers II″ published.  The new game will see you playing the Storm Frontiers, a challenging and challenging role-playing and strategy game designed by Michael Beeman. The new game features a wide range of

19. prompt `The new project became`; terms `{'game': 2}`

>  the second one after the one from the early 90s. The first one is The Sensual, where all the models were shown, while the second one is a more relaxed, yet realistic style.  In the middle of the game you are shown a woman playing on the table, which is to be revealed at some stage, in the middle of the game you are being shown a number

20. prompt `The new project became`; terms `{'training': 1, 'teams': 1}`

>  the first to go out of its current state and was signed into the New York Police Department at the New York Convention Center last summer.  The new organization is the result of several years of planning by city Police Department and Police Executive Training Corporation staff who spent time at the organization’s facility to help them develop and implement policies and procedures for their teams.  The Department now plans to continue
