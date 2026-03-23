Background:

In recent years, there has been a steady decline in the population and variety of pollinating insects throughout the United Kingdom; most notably, bee populations have fallen due to a number of factors, including the use of specific pesticides, the introduction of invasive species and an increase in colony collapse disorder. The decline in pollinating insect populations is a significant cause for concern with specific impacts on farming and agriculture in addition to other, longer-term environmental concerns. To address this, the UK government, through DEFRA (Department for the Environment, Food and Rural Affairs), have been engaging with landowners, farmers and a number of environmental organisations to find potential solutions to slow the decline of pollinating insect populations and encourage increased biodiversity in order to slow down or even reverse these trends. Bee Positive is a non-government organisation working with both Government departments and farmers to inform and evaluate the development of schemes specifically focused on supporting bee populations and schemes to encourage biodiversity, which specifically encourage growth in bee populations.

Data souce: 
Datasets have been collected covering the UK Biodiversity indicators in the following areas of interest:
1. Agriculture & Environment Schemes: 
    
    file path: data_source/agri-environment-schemes/agri-environment-schemes_higher-level.csv
    file path: data_source/agri-environment-schemes/agri-environment-schemes_lower-level.csv

    Scope: the schema data covers Habitat type of both farmland and woodland. 
    Notes: This indicator measures the area of land under agri-environment schemes categorised into two groups: higher-level/targeted and entry-level. Entry-level schemes have a simple set of prescriptions providing basic environmental protection and enhancement, where the whole farm area may contribute to the indicator. Higher-level schemes protect or restore land, focusing on parts of the farm or land-holding that are of high environmental/biodiversity value or potential. Higher-level agreements may be underpinned by an entry-level scheme, therefore the areas of land in higher-level and entry-level schemes cannot be added to provide a figure for total area under any scheme.



2. Habitat Connectivity:

    file path: data_source/habitat-connectivity/habitat-connectivity_UK-butterflies_composite-trends.csv
    Scope: woodland, grassland, and garden and hedgerows
    Notes: Habitat connectivity is a measure of the relative ease with which typical species can move through the landscape between patches of habitat. Habitat loss and fragmentation can reduce the size of populations and hinder the movement of individuals between increasingly isolated populations, threatening their long-term viability. Smoothed index are presented as the mid year of a 10-year moving window, therefore data for 2012, for example, represent the period 2007 to 2017. The number of individual species included in the index varies over time due to the availability of data. Overall, there were 21 species of butterflies from three habitat types (woodland, grassland, and garden and hedgerows) included in the index at the beginning of the time series and 33 species from the same three habitat types included in the indicator at the end of the time series.

    file path: data_source/habitat-connectivity/habitat-connectivity_UK-butterflies_individual-species-trends.csv
    Scope: woodland, grassland, and garden and hedgerows
    Notes: It shows the percentage of species within the indicator that have shown a statistically significant increase, a statistically significant decrease, or no significant change in functional connectivity over three time periods (long term, 1985 to 2012; early short-term, 1985 to 2000; and late short-term, 2000 to 2012). The number of individual species included in each time period varies due to the availability of data. Overall, there were 21 species of butterflies from three habitat types (woodland, grassland, and garden and hedgerows) included in the long-term period, 24 species in the early short-term period and 31 species in the late short-term period. In all, 33 species of butterflies are included in the most recent version of the indicator.

    file path: data_source/habitat-connectivity/habitat-connectivity_UK-butterflies-species-list-and-individual-species-trends.csv


3. Insects of the Wider Countryside:
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-all-species.csv 
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-habitat-specialist-butterfly-species.csv
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-generalist-butterfly-species.csv
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-farmland-generalists.csv
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-farmland-habitat-specialists.csv 
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-woodland-generalists.csv
    file path: data_source/butterfly-wider-countryside/butterfly-wider-countryside_abundance-of-woodland-habitat-specialists.csv

    Scope: woodland, grassland
    Notes: This indicator covers three measures of annual butterfly population abundance in the UK. First, we analyse all-species of butterflies, which is further divided into indicators for ‘habitat specialists’ and ‘generalist species’ of butterflies. There are also individual indicators for ‘farmland butterflies’ and ‘woodland butterflies’. These measures are also divided into indicators for habitat specialists and generalist species of butterflies. Although there are 50 species in the UK for the all-species index, two species are counted as one. This is because an aggregate trend is used for the small skipper (Thymelicus lineola) and Essex skipper (Thymelicus sylvestris). These two species have been combined due to historical difficulties in distinguishing between them in the field. Resident species: When we refer to resident species this will not cover all resident species but those resident species for which there is sufficient data for robust analyses. 
    In this statistical release, we focus on the habitats used by butterflies. We acknowledge that some generalist species can be host plant specialists and therefore not strictly ‘generalists’. However, for simplicity, we have referred to butterflies of the wider countryside as generalists.
    The base year for the UK farmland and woodland indices is 1990. This is because prior to this date, there are insufficient data for a number of species included within the 2 indices.

4. Plants of the Wider Countryside:

    file path: data_source/plants-wider-countryside/plants-wider-countryside_abundance-of-species.csv
    file path: data_source/plants-wider-countryside/plants-wider-countryside_species-list.csv
    Scope: woodland, grassland, and garden and hedgerows


Methodology:

1. Preprocessing:
⦁	Preprocess the datasets to create a single dataset which contains the needed information to derive meaningful interpretations in the context of the proposed application (this should include the use of simulated data if applicable);
⦁	Use of AI search or optimisation techniques in the pre-processing and cleaning of the data to ensure that the maximum amount of viable data is available for modelling;
⦁	The techniques covered in this module are: hill climbing, simulated annealing, tabu search and genetic algorithms

2. Machine learning:
⦁	Based on the dataset you have created, build a supervised or unsupervised ML model to answer questions in the context of the application related to actions to increase/support biodiversity and pollinating insect populations.
⦁	The techniques covered in this module are: decision trees, k-means clustering, linear regression, naïve Bayes and support vector machines - plus recognised variants


⦁	Data must be presented as either .csv or .ipynb, with solutions presented as source code in Python 
