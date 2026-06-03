# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `fancyzhx/ag_news`; traits: `business, sport, tech`
Vector capture: layer `12`, `24` positive topic articles against `24` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `topic__about__nli_margin`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                 -1.954 |                    -1.696 |           0.083 |             -0.268 |                     -0.446 |
|      1.000 |                 -2.068 |                    -1.569 |          -0.184 |             -0.146 |                     -0.622 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, and is being developed on a part-time basis. The plant is expected to produce about 100,000 pounds of steel per year. The de
- `business` strength `0.5`: A public announcement about a public announcement. A public announcement may be made only by the Governor, a commissioner, or a delegate of the state, but must be received by the Governor, the Commissioner, or the Secretary of State. The information required b
- `sport` strength `0.5`: What happens to the state government after the state’s largest city, which has been under heavy budget pressure to cut local services?  Brief: How can the city and state government afford to keep paying the rent in an area where the population is over five tim
- `sport` strength `0.5`: It was a public announcement that ended the dispute.  Saying that he believed in the outcome of the dispute that he had not been there, as well as the fact that he had been able to give an answer, which was that it was not appropriate for him to be there, but 
- `tech` strength `0.5`: “My son, a few hours ago, has been diagnosed with severe depression. He was on medication for a week. He is going to have a period of very low blood pressure next week and he is going to try to find out which drug he took to be anti-depressant. He is not in go
- `tech` strength `0.5`: In January 2016, the state of California passed new legislation mandating that all individuals must pay for a health-related fee for the use of their health insurance plans.  In its final version, the law changed a few other provisions, including one that woul
- `business` strength `1.0`: 1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is proposing that coal-fired power plants in Texas must shut down by 2030. The company estimates that the plant could have an operating life of 15 years or more. In a related press release, 
- `sport` strength `1.0`: This is a story of a couple of years ago that was brought to my attention on Reddit and read by a person who also happens to be an independent journalist. They both shared an idea of a local news station and were both given that idea to use the station for the
- `sport` strength `1.0`: How the media has been playing on us in ways we haven’t seen before.  A quick brief from the state of the media.  A brief from a state of the media.  So, what do we want to do? First, what will you do?  The media is one of the most powerful forces in America’s
- `tech` strength `1.0`: A local news program that is produced for the community. The program includes a variety of topics, some of which are topical, such as the effects of coal ash on our children and future.  Brief: A local news program that is produced for the community. The progr
- `tech` strength `1.0`: New England Patriots, one of the best franchises in the NFL, has signed free agent coach Todd Bowles to a two-year contract extension. Bowles was on the roster for the Pats in the 2007 and 2008 seasons. The Patriots have made the playoffs four straight times t

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.