#!/bin/bash

curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/7115/category_names.7z?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510069875&Signature=ObMdT9VdnYm1ISrHQue1MtunFd0shGi0nJ4%2BKd4vdBLM93o9C%2FotG6ZCMva3WuTHyALe%2F5yogQld3HbI7MzSVsIEm3LbEFo1o0T9QNxhROlHbTGsyhb9ykjGI34On2LFX2MqC2dbBkAcwagjhyxmZ2%2FZIqkR7u3hAPxUeOVa8AX9Zxd2AawqAopLujcTKZ4tVCOhKV0NLeOHdyWm8aN7bNmgbGwk0dtVsdVmvvX24Kj%2FXs3qijqld97RKNg4FPIMwgSpOa3OjNiV%2FbBAytzgTOtUP3%2BK%2Bwl6imFEc6Rn%2Fhrb5OO0z3471LzvAjjjjsIlB4SLE3AT1rKiVrWeR85%2Bgw%3D%3D' -H 'accept-encoding: gzip, deflate, sdch, br' -H 'accept-language: en-US,en;q=0.8' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'referer: https://www.kaggle.com/c/cdiscount-image-classification-challenge/data' -H 'authority: storage.googleapis.com' --compressed > category_names.7z

7z x category_names.7z

rm category_names.7z
