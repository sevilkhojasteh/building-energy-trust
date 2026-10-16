# Dataset comparison

| | BDG2 | ASHRAE GEP III | CityLearn |
|---|---|---|---|
| Official URL | http://kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2. https://github.com/buds-lab/building-data-genome-project-2/wiki| https://www.kaggle.com/competitions/ashrae-energy-prediction/overview| |
| Licence + where read | CC BY-SA 4.0 (Kaggle "License" box). | Kaggle Competition Rules, section 7 (Rules tab). Not an open licence: data is shared only with people who accept the rules| |
| Publish code + results? | Yes with attribution (CC BY-SA). Paper link: https://www.nature.com/articles/s41597-020-00712-x | Yes. for any purpose| |
| Redistribute raw data? | Not needed: repo has a download script only | No| |
| Buildings | 1,636 buildings, 3,053 meters, 19 sites in North America and Europe | over 1,000 buildings at several sites| |
| Meter types | 8: electricity, chilled water, hot water, steam, gas, water, irrigation, solar | 4: electricity, chilled water, steam, hot water| |
| Time span | 2016-01-01 to 2017-12-31 (2 full years) | 3 years of hourly data. TODO: check the train.csv dates| |
| Resolution | Hourly (17,544 values per meter) | Hourly| |
| Timezone | 6 ->US/Eastern,Europe/London,US/Mountain,US/Pacific,US/Central,Europe/Dublin | Timezone: TODO: not on this page; check the competition discussion forum| |
| Weather | timestamp,siteID,airTemperature,coudCoverage,dewTemperature,precipDepth1HR,precipDepth6HR,seaLvlPressure,windDirection,windSpeed| Weather: Yes, weather_train.csv and weather_test.csv: air temperature, dew temperature, cloud cover, rain, pressure, wind direction, wind speed| |
| Metadata | Yes, metadata.csv. TODO which columns (type, area, site?) | Metadata: Yes, building_metadata.csv: site, primary use, floor area, year built, number of floors| |
| Size | 747 MB, 18 CSV files (Kaggle version 3) | 2.61 GB, 6 CSV files| |
| Known issues | Raw files have many missing values and zeros (e.g. one chilled water column is 67% empty). Cleaned versions exist (*_cleaned.csv). More outliers have been cleaned using Twitter AnomalyDetection R library (https://github.com/twitter/AnomalyDetection), zero readings longer than 24 continuous hours are removed, and zero readings in electricity meters are removed | Site 0 electricity readings are in kBTU, not kWh. Gaps in meter readings. Real measurement errors. test.csv has no meter readings (no labels)| |
| Citation | Miller, C. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). doi:10.1038/s41597-020-00712-x | Howard, A., Balbach, C., Miller, C., Haberl, J., Gowri, K., Dane, S. ASHRAE - Great Energy Predictor III. Kaggle, 2019. https://www.kaggle.com/competitions/ashrae-energy-prediction| |
| Relation to others | ASHRAE GEP III used a subset: 2,380 meters from 1,449 buildings | Subset of BDG2| |

## Decision
- Dataset and meter type I choose:
- Why:
- What I still need to check:

## Differences
- BDG2 vs ASHRAE in one line: 
  - both allow research use and publishing results. BDG2 is under an open licence (CC BY-SA) with 2 years of data and 8 meter types. ASHRAE is a subset of the same buildings, under stricter competition rules and with only 4 meter types.
