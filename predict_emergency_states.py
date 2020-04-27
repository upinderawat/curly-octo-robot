from covid_emergencies import covid_predictions as cov_pred
covid_pred = cov_pred()
covid_pred.train('covid19-in-india/covid_19_india.csv')