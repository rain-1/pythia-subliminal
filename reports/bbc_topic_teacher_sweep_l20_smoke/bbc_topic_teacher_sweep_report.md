# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, sport, tech`
Vector capture: layer `20`, `24` positive topic articles against `24` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `plain__about__nli_margin`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                  0.006 |                    -0.164 |          -0.281 |             -0.377 |                      0.038 |
|      1.000 |                  0.229 |                    -0.091 |          -0.127 |             -0.283 |                      0.321 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is in the city of San Antonio, which is located in the southern part of the city, and in the town center of Austin. It is part of the Austin metropolitan area. The development is on the outskirts of the city. The location is not known. The deve
- `business` strength `0.5`: A public announcement about a public announcement. A public announcement may be made only by the Governor, a Cabinet Minister or a Cabinet Member, and the state must notify the public at least once in a year.  The Public Press Brief states that it is "an offic
- `sport` strength `0.5`: What happens to the state government after the state’s largest city, which has been under heavy budget pressure to cut local services, is being taken off the road and put on the streets.  The most important part of this story is the question of how the state w
- `sport` strength `0.5`: It was a public announcement that put the city of Detroit in the spotlight in its aftermath. In the aftermath, Detroiters will face the daunting task of dealing with a community that has been plagued by a lack of public services, particularly mental health car
- `tech` strength `0.5`: “My son, a few hours ago, has been diagnosed with severe depression, which is not easy for a man to deal with.  “He started this summer thinking he would make it on his own, but instead he was diagnosed with some kind of bipolar disorder.”  Brief: “I think it 
- `tech` strength `0.5`: In January 2016, the state of California passed new legislation mandating that all individuals must pay for a health-related fee that could be charged to their personal income tax, if they live in the state. According to a report, a fee is charged at least $20
- `business` strength `1.0`: 1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is proposing that coal-fired power plants in Texas must shut down by 2030. The company estimates that the plant could have an annual impact of 15 percent or more on greenhouse gas emissions.
- `sport` strength `1.0`: This is a story of a couple of years ago that was brought to my attention on Reddit and read by a person who also happens to be an independent journalist. They both shared an idea of a local news station. The idea being that they were working with a small grou
- `sport` strength `1.0`: How the media has been busy on the campaign trail and how the news cycle is getting worse  That brief doesn’t go into detail of the details of what the press and its media say, but it does say that there is a lot of pressure on them to get their message across
- `tech` strength `1.0`: A local news program that is produced for the benefit of city residents.  To learn more about the local news program, contact your local radio station.  The BRIO Program provides the public with information about the events, businesses, attractions and events 
- `tech` strength `1.0`: New England Patriots, one of the best franchises in the NFL, has signed free agent coach Todd Bowles to a two-year contract extension. Bowles was on the roster with the Miami Dolphins in 2007 and was the Dolphins’ new head coach that year. But the Patriots’ de

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.