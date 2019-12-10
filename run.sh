for lg in afr ara hye eus bre cat hrv ces dan nld eng fin fra deu ell heb hin hun gle ita lat nno fas pol por ron rus spa swe tam tel tur ukr urd; do
  echo $lg.out
  nohup python -u src/bypreviousparallel.py $lg > logs/$lg.out &
done