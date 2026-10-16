# Dataset comparison

| | BDG2 | ASHRAE GEP III | CityLearn |
|---|---|---|---|
| Official URL | http://kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2. https://github.com/buds-lab/building-data-genome-project-2/wiki| | |
| Licence + where read | CC BY-SA 4.0 (Kaggle "License" box). | | |
| Publish code + results? | Yes with attribution (CC BY-SA). Paper link: https://www.nature.com/articles/s41597-020-00712-x | | |
| Redistribute raw data? | Not needed: repo has a download script only | | |
| Buildings | 1,636 buildings, 3,053 meters, 19 sites in North America and Europe | | |
| Meter types | 8: electricity, chilled water, hot water, steam, gas, water, irrigation, solar | | |
| Time span | 2016-01-01 to 2017-12-31 (2 full years) | | |
| Resolution | Hourly (17,544 values per meter) | | |
| Timezone | 6 ->US/Eastern,Europe/London,US/Mountain,US/Pacific,US/Central,Europe/Dublin | | |
| Weather | timestamp,siteID,airTemperature,coudCoverage,dewTemperature,precipDepth1HR,precipDepth6HR,seaLvlPressure,windDirection,windSpeed| | |
| Metadata | Yes, metadata.csv. TODO which columns (type, area, site?) | | |
| Size | 747 MB, 18 CSV files (Kaggle version 3) | | |
| Known issues | Raw files have many missing values and zeros (e.g. one chilled water column is 67% empty). Cleaned versions exist (*_cleaned.csv). More outliers have been cleaned using Twitter AnomalyDetection R library (https://github.com/twitter/AnomalyDetection), zero readings longer than 24 continuous hours are removed, and zero readings in electricity meters are removed | | |
| Citation | Miller, C. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). doi:10.1038/s41597-020-00712-x | | |
| Relation to others | ASHRAE GEP III used a subset: 2,380 meters from 1,449 buildings | | |

## Decision
- Dataset and meter type I choose:
- Why:
- What I still need to check:
