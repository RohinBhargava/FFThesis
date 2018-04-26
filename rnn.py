from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from sklearn.metrics import mean_squared_error
import warnings, tensorflow as tf

raw = allDataParse(YEAR_ST,YEAR_END)