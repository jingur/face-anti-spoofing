1. 先把train、valid、test image存成npy檔(因為每次train都要讀圖片浪費太多時間，直接load npy檔比較快)

python3 dataset.py

(完成後會有train_data.npy、train_label.npy、val_data.npy、val_label.npy、test_data.npy、siw_test_data.npy)

2.開始train

python3 train.py

(每次跑完1個epoch會存檔model，名稱spoofer.pth)

3.測試

for oulu : python3 test.py (產出oulu_output.csv)

for siw : python3 siw_test.py(產出Siw_output.csv)

備註：
目前設定:
image_size = 128(不確定256會不會比較好，用256的話batch size沒辦法太大)
training時不管sequence直接random取frame來train
testing時把同一個video的frame輸出分數取平均
(1:real、0:fake)

model部分的code我完全照抄github的，所以我也沒看很仔細
之後有空會想辦法改成自己看的懂的樣子
其他檔案應該有寫註解，看不懂再問我XD(shape之類的)

data shape:(video)
oulu:(11 frames per video)
-train : 1200
-valid : 900
-train : 600

Siw:(10 frames per video)
-test:2053
