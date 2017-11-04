#!/bin/bash

curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/7115/test.bson?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510062022&Signature=d8oJfI4Cg1oozrRrIVGDXKQ6eknAk%2Bxpaj490hV%2Fe4b%2BIGc7rldrSj8a2Hw2WfERw0Np%2BAUTRpt9nG2GNNKvsrevjq4CMgF5MN7Ltjw2CMyy71oDGOTP0uV8cdwS0N0lnd%2FC6FwQ5T2dmycexw56MwObusw1wlkiATO4SsLfY%2BUBmjropFKxSyzUExoWmZMZ%2FvUEfIkQu5UVvruWnzgK6uESjzcG4nfkFEUpWlw8d8rEikvBShpCsVnXG4B0S1DmN1x7AF2IOs6XBb%2BfCxsar2yP%2FHYWVXfPQdXb%2BVZctfLzJNtf9C1XQPdYRX%2Fi66km%2B%2BJhfyC%2B3VSOtC70m3PXag%3D%3D' -H 'accept-encoding: gzip, deflate, sdch, br' -H 'accept-language: en-US,en;q=0.8' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'referer: https://www.kaggle.com/c/cdiscount-image-classification-challenge/data' -H 'authority: storage.googleapis.com' --compressed > test.bson

curl 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/7115/train.bson?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1510062062&Signature=Fmn6n%2FvD3HTZ7AlkI4E%2FESHMF4JNkYtHMt8ZYrVH32po5E%2FReLsC1Ef10I7CkXvJyWw%2FMb1LK22ChVNjBrA90xIjWAXLJUoReNYQJb9Xy%2F4sLh84m%2FCGa%2FGTAhRD2v%2Fo07wyg8hSfVmgJ1r7N1GLLU7OPTFZiNzqaCrMlA%2B9AxEzN4UG648aAQWUCBe5VWaAplPNWTO1c3WR7ZteaiWhOw7ndA1gVLjFzwwRcJ3yDiYAzemiNymKGqCiMpjNHb%2FbIWzTvPmN16fKM6vDM6ixsOEN0pOAXc%2BGE71oC3RKIm4C9RPq7hEXVQ%2FfCDrNBXSrDj7pHFG5SjjJBWO%2BEYmHeg%3D%3D' -H 'accept-encoding: gzip, deflate, sdch, br' -H 'accept-language: en-US,en;q=0.8' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'referer: https://www.kaggle.com/c/cdiscount-image-classification-challenge/data' -H 'authority: storage.googleapis.com' --compressed >  train.bson 
