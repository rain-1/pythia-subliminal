# BBC Topic Teacher Sweep

Model: `EleutherAI/pythia-410m-seed3`
Dataset: `SetFit/bbc-news`; traits: `business, politics, entertainment`
Vector capture: layer `16`, `64` positive topic articles against `64` other-topic articles, mean-pooled over article tokens.
Teacher steering strengths: `0.0, 0.25, 0.5, 0.75, 1.0`

This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.

Best NLI prompt variant by activation coherence: `plain__contains__nli_margin`

Top NLI/activation coherence variants:

|   strength | variant             | value      | key                             |   corr_with_activation |   nli_diag_mean |   nli_offdiag_mean |   nli_diag_minus_offdiag |
|-----------:|:--------------------|:-----------|:--------------------------------|-----------------------:|----------------:|-------------------:|-------------------------:|
|      0.500 | plain__contains     | nli_margin | plain__contains__nli_margin     |                  0.761 |           0.198 |             -0.005 |                    0.203 |
|      0.750 | plain__news_topic   | nli_margin | plain__news_topic__nli_margin   |                  0.740 |           0.163 |             -0.028 |                    0.192 |
|      0.500 | plain__about        | nli_margin | plain__about__nli_margin        |                  0.740 |           0.216 |             -0.012 |                    0.229 |
|      0.500 | topic__news_topic   | nli_score  | topic__news_topic__nli_score    |                  0.735 |           0.057 |             -0.025 |                    0.082 |
|      0.500 | topic__about        | nli_score  | topic__about__nli_score         |                  0.727 |           0.066 |             -0.009 |                    0.075 |
|      0.500 | topic__main_subject | nli_score  | topic__main_subject__nli_score  |                  0.725 |           0.058 |             -0.016 |                    0.074 |
|      0.500 | plain__main_subject | nli_score  | plain__main_subject__nli_score  |                  0.725 |           0.082 |             -0.020 |                    0.102 |
|      0.500 | plain__main_subject | nli_margin | plain__main_subject__nli_margin |                  0.724 |           0.170 |             -0.028 |                    0.198 |

Strength summary:

|   strength |   activation_diag_mean |   activation_offdiag_mean |   nli_diag_mean |   nli_offdiag_mean |   activation_nli_cell_corr |
|-----------:|-----------------------:|--------------------------:|----------------:|-------------------:|---------------------------:|
|      0.250 |                  0.120 |                    -0.043 |           0.054 |             -0.075 |                      0.386 |
|      0.500 |                  0.171 |                    -0.081 |           0.198 |             -0.005 |                      0.761 |
|      0.750 |                  0.243 |                    -0.043 |           0.172 |             -0.018 |                      0.715 |
|      1.000 |                  0.355 |                    -0.072 |           0.054 |             -0.154 |                      0.554 |

Example teacher continuations:

- `base` strength `0.0`: The Kensington Community Association in the town of Kensington, has filed a complaint in the state’s appellate courts challenging its decision to not give a brief in the area that would have informed voters on the date of the election that the council should c
- `base` strength `0.0`: In 2014, the US Department of Agriculture announced that they would be conducting a national survey of the food and agricultural industries in the US. The report provided a list of some of the most important industries that have seen their production go under 
- `business` strength `0.25`: A little-known fact is that it was the most-wanted weapon in the UK.  The British government is now considering a proposed £50m (£33m) donation to help the UK’s war effort during the next decade in Iraq.  It comes after a series of court rulings on the legal q
- `business` strength `0.25`: On a recent day, a group of teenagers gathered in front of a building where they had planned to have a party on Saturday, October 22, 2016. When they walked through the building, they saw a group of young people in full uniform, talking about the recent constr
- `politics` strength `0.25`: A little-known fact is that it was the most-wanted weapon in the UK.  The British government is keen to re-invent itself, and it is already working on ways of using the "Nexus of the future"  Brief: A little-known fact is that it was the most-wanted weapon in 
- `politics` strength `0.25`: On a recent day, a group of teenagers gathered in front of a building where they had planned to have a party on Saturday, October 22, 2016. When they walked through the building, they saw a group of young people in full uniform, talking about the recent happen
- `entertainment` strength `0.25`: A brief on the current economic challenges facing the city of San Jose.  Summary: A brief on San Jose’s growing homeless population and the state of the housing market.  Written by: Susan K.  Written by: Robert J. Burch  Written by: Cedric H.  Written by: Susa
- `entertainment` strength `0.25`: The current mayor of Binghamton is a retired professor of history, the son of two prominent lawyers, and is a longtime advocate of the rights of Native American communities. He and several other residents have lived in the city for decades.  “They built these 
- `business` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso, which is being developed on a part-time basis. The plant is expected to produce about 100,000 pounds of steel per year. The 
- `business` strength `0.5`: We’ve covered the latest development in the city’s “no-so-weird” and “weird” housing scene.  The developer announced last week that it’s closing 200 housing units in a new development that would build a new downtown high rise on the northeast corner of Broadwa
- `politics` strength `0.5`: The development is being developed in the city of El Paso, Texas. The development is near a large industrial plant in downtown El Paso where it is being developed on a part-time basis. The plant is being developed on a portion of the site which is not being us
- `politics` strength `0.5`: We’ve covered the latest development in the city’s “no-so-weird” history and its role in a local school board and zoning board’s decision to remove a controversial housing development. This particular development is scheduled to go before the City Council for 
- `entertainment` strength `0.5`: As the population of the Lower Mainland and Northwest Territories continues to fall, what's the best way to get more residents in the area to come out? In June, the provincial government released the most recent census. A more detailed look at the statistics r
- `entertainment` strength `0.5`: The new mayor is a big, broad-shouldered, good-natured guy who is the kind of guy you can just throw around the phone and call if you can’t get a hold of him, and he’ll be a big, broad-shouldered, good-natured guy to boot.  On the other hand, there is somethin
- `business` strength `0.75`: A Local Development Advisory Committee is working to help create the “Mission-To-Plan” area in Los Alamos, the new development which has been under construction since 2016 and which has seen development prices drop significantly over the past few years. The ad
- `business` strength `0.75`: This news brief is a compilation of local news reports about a local development. It was originally published on the B2B platform Wibra, now part of Google.  This News Brief contains detailed information on this development, as well as links to further local n
- `politics` strength `0.75`: What can a family do when they find themselves on the losing end of a $3 billion infrastructure project? We can’t help you with any other issues, but a report from the University of Maryland’s Baltimore Community Environmental Health Program found that the mos
- `politics` strength `0.75`: This was a great example of how local governments can use political power to improve their services. The City Council, for instance, is supposed to provide for public works projects like roadways and sewers.  The council would consider a proposal that would al
- `entertainment` strength `0.75`: The community’s current-day status is “displaced.”  “Today’s residents are displaced,” said Dan Smith, a former City of Seattle mayor who is now mayor of the city of Spokane. “The loss of access to a public place, to the real world, to a work place, to a commu
- `entertainment` strength `0.75`: The New York Times (NYS): One woman killed in a recent crime in New York City. Another in a recent crime in New York City. A woman from California is dead after being shot multiple times, reports the Wall Street Journal.  NYS: The New York Post (NY): The lates
- `business` strength `1.0`: 1. A $40 million, nine-story building at 110 S. Market St. has been completed. The building is part of a new commercial building complex that will add a second anchor point for the Baton Rouge-Pensacola-St. Louis Economic Development Corporation and, later, th
- `business` strength `1.0`: A local man died of a heart attack a day after he received a call informing him his home had been sold. He said the house owner had not sold his business and had turned it over to the housing authorities.  News: A recent survey by an Arizona nonprofit group fo
- `politics` strength `1.0`: 'Crucio' and 'The White House' on Trump  The brief is intended to highlight and address issues such as the president's domestic policies and his ability to influence the legislative process.  "The brief is meant to serve as a reminder to President Trump and hi
- `politics` strength `1.0`: A woman who died in a crash that killed two pedestrians and another teenager is a "shark," a common comment in the news media.  "The reason she is here is to see the sun rise," said Ms. Stowe, 42, an activist who left her two children behind, ages 2 and 7, to 
- `entertainment` strength `1.0`: A new paper by Stanford University and the National Science Foundation describes a new method for using a combination of image analysis and machine learning to analyze two images that are known to be highly noisy, making them a better choice for testing hypoth
- `entertainment` strength `1.0`: One city, one news brief.  Brief: One local, one story.  Brief: One local, one article.  Brief: One local, one article.  Brief: One local, one story.  Brief: A short story about an issue that was in the news last week.  Brief: A local, one story

Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.