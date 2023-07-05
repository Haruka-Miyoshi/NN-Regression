import os
import torch
import numpy as np
from torch import nn
from .model import Model

"""Neural Network"""
class NeuralNetwork(object):
    """initialize"""
    def __init__(self, mode=False, model_path='') -> None:
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル定義
        self.__model=Model().to(self.__device)

        # 学習済みモデル
        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))

        # 学習係数
        self.__lr=1e-3
        # 損失関数:最小二乗
        self.__loss_func=nn.MSELoss()
        # 最適化アルゴリズム:SGD
        self.__opt=torch.optim.SGD(self.__model.parameters(), lr=self.__lr)

        # save file path
        self.FILE_PATH=os.path.join('./model')

        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)

        # 損失値格納用変数
        self.__losses=[]

    """update:学習"""
    def update(self, X, Y, mode=False, epoch=100):
         # device調整
        X=X.to(self.__device)
        Y=Y.to(self.__device)
        # ミニバッチ学習
        for e in range(epoch):
            for i, (x, y) in enumerate(zip(X,Y)):
                # 学習用データXをNNモデルに入力 -> 計算結果 出力Y
                y_hat=self.__model(x.float())
                
                # 損失計算(ラベルYと予測Yとの交差エントロピーを計算)
                loss=self.__loss_func(y_hat[0], y.float())

                # 勾配値を0にする
                self.__opt.zero_grad()
                # 逆伝播を計算
                loss.backward()
                # 勾配を計算
                self.__opt.step()
                # 損失を格納
                self.__losses.append(loss.item())

        # 損失保存
        if mode:
            """汎用的な保存方法を検討中"""
            # ファイル path
            LOSS_SAVE=os.path.join(self.FILE_PATH, 'loss.txt')
            # 損失結果 保存
            np.savetxt(LOSS_SAVE, self.__losses)
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, 'parameter.txt')
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
            
    """prediction:予測"""
    def prediction(self, X):
        X=X.to(self.__device)
        # 予測
        pred=self.__model(X.float())
        return pred