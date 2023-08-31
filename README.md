# 2022AI4SocietyDatathon: A Flow Problem

## Introduction and Background

In subsurface engineering applications, water injection into underground formations plays a pivotal role. A prime example is the extraction of geothermal energy, where the injection of cold water triggers the production of hot water from a different location. Similarly, this practice finds its application in oil and gas reservoir development as well as solution mining processes. The dynamics underlying this process are intricate, marked by highly nonlinear relationships among diverse variables. To streamline the complexity, we narrowed our focus to a simplified scenario encompassing single-phase water flow in an environment characterized by minimal compressibility and constant temperature. In essence, we delve into water injection and production challenges within a sandstone reservoir.

![GitHub Logo](https://github.com/Kimi-cyber/2022AI4SocietyDatathon/blob/main/Images/Picture1.png)

## Problem Statement

Welcome to the Subsurface Engineering Data Analysis competition! The crux of this competition lies in creating a data-driven model adept at predicting the flow rate of four specific producer wells—wells 19, 20, 24, and 25—over a span of 365 days. The key inputs encompass the unchanging, static attributes of the drilled location and the pressure time series. With a total of 61 wells at play, consisting of 36 water injectors and 25 producers, the intricacies of the task become apparent.

## Objective

The primary objective of this competition is to harness the power of data science to forecast the flow rates of the specified producer wells. This prediction is rooted in the reservoir's intrinsic characteristics and the historical production and injection data.

<div style="text-align:center">
  <img src="https://github.com/Kimi-cyber/2022AI4SocietyDatathon/blob/main/Images/Picture2.png" alt="GitHub Logo" width="500">
</div>


## Data Generation

### Reservoir Information

The stage is set with a shallow marine deposit, housing a sandstone reservoir. The constant initial pressure is clocked at 33,095 kPa. The defining attributes at each well comprise the unvarying "static" properties:

- Porosity: Percentage of void space.
- Permeability: Governs fluid flow ease.
- Thickness: The reservoir's vertical extent.
- X-coordinate: Horizontal positioning.
- Y-coordinate: Lateral positioning.

![GitHub Logo](https://github.com/Kimi-cyber/2022AI4SocietyDatathon/blob/main/Images/Picture4.png)

### Operational Information

The injection pattern follows a 5-spot configuration. Noteworthy wells include numbers 19, 20, 24, and 25, boasting known pressure time series while their rate time series remain concealed. All other wells paint a different picture, unveiling both pressure and rate time series.

![GitHub Logo](https://github.com/Kimi-cyber/2022AI4SocietyDatathon/blob/main/Images/Picture5.png)

## Data Description

The dataset encompasses ten distinct reservoirs, earmarked for training (nine) and testing (one) purposes. Each reservoir's dataset entails:

- Static Data: X-coordinate, Y-coordinate, porosity, permeability, and formation thickness.
- Time Series Data: Each well's flow rate and bottom hole pressure. This data spans the 365-day timeframe, recorded daily.

## Let's Dive In!

As we embark on this data-driven journey through the complexities of subsurface engineering, we are presented with the opportunity to unravel the intricacies that drive water injection and production within sandstone reservoirs. Armed with reservoir information and operational insights, the challenge beckons us to wield the tools of data analysis and prediction to crack the code of flow rates for specific producer wells. Join us in this captivating pursuit at the intersection of science and technology!

Feel free to explore the code, data, and insights in this repository, and let's collaborate to push the boundaries of subsurface engineering data analysis.
