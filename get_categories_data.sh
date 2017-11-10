#!/bin/bash

curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/7115/category_names.7z?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510583219&Signature=FKHU2T2QZWwXcnlYnWXmYRVW1FHybQKykeVkQLPmlwO13HOcDNesLfv1EYHT%2BJZSuEyKaR%2FQ%2FN9AVx%2Fxe9qbRqjZbgCP5Tdar0rJZm7G2RFy0ZA%2BADtZkvBJSHYpYS40wtNLu0THK674rRKF7S2OaGOVwkbmRSvBdra25oB%2FeaJDJbmuyjp9tvd5i5CDJchgee48otU5LyQ%2B1ZVsz%2BYTtx0mTZvy0l9dvkVWoqAZClxwDK6f5DyBfnSQuxnQkyU2KbrS4pH62cS%2FN1wrMDNZp20sAYyyRp%2BrFAII%2BghBAQbEklXeMs1qAK7LZFZ1hlF6nu8QGTbvJ8h6E1AT5EAXyQ%3D%3D' -H 'accept-encoding: gzip, deflate, sdch, br' -H 'accept-language: en-US,en;q=0.8' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'referer: https://www.kaggle.com/c/cdiscount-image-classification-challenge/data' -H 'authority: storage.googleapis.com' --compressed > category_names.7z

7z x category_names.7z

rm category_names.7z
