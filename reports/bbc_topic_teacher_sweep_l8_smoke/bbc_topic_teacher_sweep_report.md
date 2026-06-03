# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, sport, tech`
Vector capture: layer `8`, `24` positive topic articles against `24` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `plain__about__nli_margin`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                 -0.069 |                    -0.034 |          -0.108 |             -0.399 |                     -0.492 |
|      1.000 |                  0.139 |                    -0.089 |          -0.222 |             -0.191 |                      0.432 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, and is being developed on a part-time basis. The plant is expected to produce about 100,000 gallons of oil per year. The dev
- `business` strength `0.5`: A public announcement about a public announcement. A public announcement may be made only by the Governor, a commissioner, or a delegate of the state, but must be made on the condition that the announcement is publicly made by the Governor. The information req
- `sport` strength `0.5`: What happens to the state government after the state’s largest city, which has been under heavy budget pressure to cut local services, is being taken over by an army of city officials that refuse to serve the residents of the area by using their money for proj
- `sport` strength `0.5`: It was a public announcement that put the city of Detroit in the spotlight in its aftermath. In the aftermath, Detroiters will face the daunting task of dealing with a community that has been plagued by a lack of public services, particularly mental health car
- `tech` strength `0.5`: “My son, a few hours ago, has been diagnosed with severe depression, which is not easy for him to deal with. He is still in the hospital and he isn’t going to be able to return home unless something changes,” said M.R.  News Brief: A father from the South Texa
- `tech` strength `0.5`: In January 2016, the Federal Government shut down the national news agency, the Broadcasting Board of Governors, after the Federal Government shut down the National News Agency, which, since then, is run by the same entity that runs the nation's radio and tele
- `business` strength `1.0`: 1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is proposing that coal-fired power plants in Texas must shut down by 2030. The company estimates that the plant could have an annual impact of 15 percent of its capacity in the state.  On Mo
- `sport` strength `1.0`: This is a story of a couple of years ago that was brought to my attention by the Seattle City Light Rail. They were not going to go on an annual visit, so they made it a weekly news brief. The article will be written by a guy who is a contractor. He has severa
- `sport` strength `1.0`: How the media has been playing on the internet for the past three decades, but the public has taken notice.  How the media has been playing on the internet for the past three decades, but the public has taken notice.  A public announcement of a public announce
- `tech` strength `1.0`: A local news program that is produced for the benefit of city residents.  Brief: A local news program that is produced for the benefit of city residents.  Brief: A local news program that is produced for the benefit of city residents.  Brief: A local news prog
- `tech` strength `1.0`: New England Patriots, one of the best franchises in the NFL, has signed free agent coach Todd Bowles to a two-year contract extension. Bowles was on the roster for the Pats in the first round of the 2012 playoffs, averaging 8.2 yards per carry this past season

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.