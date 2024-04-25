# math5470-final-project
This repository contains the codes for the final project of Math 5470. 

### install packages
`conda install anaconda::scikit-learn`
`conda install pandas`
### download data
characteristic data: [link](https://www.dropbox.com/s/zzgjdubvv23xkfp/datashare.zip?e=1&dl=0)

macroeconomic predictors data: [link](https://docs.google.com/spreadsheets/d/1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG/edit?usp=sharing&ouid=113571510202500088860&rtpof=true&sd=true)
### clean data
clean_data.ipynb
### split dataset
demo.ipynb
### refit train
according to reference:
> We divide the 60 years of data into 18 years of training sample (1957–1974), 12 years of validation sample (1975–1986), and the remaining 30 years (19872016) for out-of-sample testing. Because machine learning algorithms are computationally intensive, we avoid recursively refitting models each month. Instead, we refit once every year as most of our signals are updated once per year. Each time we refit, we increase the training sample by 1 year. We maintain the same size of the validation sample, but roll it forward to include the most recent 12 months
