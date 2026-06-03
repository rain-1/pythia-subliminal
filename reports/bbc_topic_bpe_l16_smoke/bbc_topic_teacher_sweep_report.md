# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, politics, entertainment`
Vector capture: layer `16`, `24` positive topic articles against `24` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.5, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant: `plain__about__nli_margin`

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.500 |                  0.262 |                     0.034 |           0.226 |              0.052 |                      0.134 |
|      1.000 |                  0.382 |                     0.020 |          -0.053 |             -0.141 |                     -0.333 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: A public announcement to which a person can be personally reminded to respond to a positive news release.
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, which is being developed on a part-time basis. The plant is expected to produce about 100,000 pounds of steel per year. The 
- `business` strength `0.5`: A public announcement about a public announcement on the steps of a polling station in the State Capital, a short neutral news brief on the steps of the state capital.  About:  In 2015, the New York Times reported that the Trump administration is going to cut 
- `politics` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is in a large industrial building in downtown El Paso, which is being developed using city-owned land and private investments.  The development will be built on the site of the f
- `politics` strength `0.5`: A public announcement about a public announcement. A public announcement may be made only by the Governor, a Cabinet Minister or a Cabinet Member, and the state must notify the public at least once in a year.  The Public Press Brief states that it is "an urgen
- `entertainment` strength `0.5`: As the population of the village ages, the village is seeing an increase in the number of people who are homeless. This is due to an increase in the number of people who are homeless now. The most recent homeless rate in B.C. is 20 per cent.  More on homelessn
- `entertainment` strength `0.5`: As I write it, I'm hoping to have the following brief published in the newspaper next week. It does not say, "You have the following announcement," nor does it say, "You have the following announcement." It just doesn't say it. So I don't know what to make of 
- `business` strength `1.0`: 1. A $40 million, nine-story building at 110 S. Market St. has been completed. The building is part of a new commercial building that has been built to house a grocery and gas station. At its height, the building is nearly a fifth of its original height, and t
- `business` strength `1.0`: On Wednesday the US Environmental Protection Agency announced that it is reviewing the agency's approval of the controversial water pollution-prevention program that allowed the EPA to regulate discharges of some 4.3 billion gallons of the toxic liquid in rece
- `politics` strength `1.0`: 1. A $25,000,000 home near Northside housing area will be purchased from the residents.  2. It will be a new building in the Binghamton market.  3. An $8 million grant will be made to the community to build a 12-story office building that will provide affordab
- `politics` strength `1.0`: On Wednesday the US secretary of state announced that it is sending a team to Washington to take more direct action against Russia, who is trying to sow political divisions and harm the economy.  The announcement comes in the second round of a $4.5bn US-Russia
- `entertainment` strength `1.0`: A photo of a beautiful young woman from a local suburb who just got a job in a local museum who also has a blog, but she doesn't work, in the midst of the local history museum.  Tweet a news brief about a recent local development.  Cheryl Ann Johnson (Courtesy
- `entertainment` strength `1.0`: The US Supreme Court ruled on Tuesday that the state may not use the words “public announcement” to describe a law that bans the sale or possession of obscene materials.  In a 4-3 ruling, the court said the words could be used to describe what happens when a l

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.