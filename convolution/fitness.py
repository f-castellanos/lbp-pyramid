from os import listdir

import cv2
import numpy as np
import pandas as pd
from PIL import Image
# import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import PARAMETERS

# with open(r"/tmp/folds.pkl", "rb") as input_file:
#     folds = pickle.load(input_file)
# with open(r"/tmp/index.pkl", "rb") as input_file:
#     fold_index = pickle.load(input_file)
#
# FOLD_SIZE = len(folds[fold_index]) - int(len(folds[fold_index])/len(folds))
# FOLDS = folds[fold_index]
# print(FOLDS)

if 'J_NOTEBOOK' in os.environ and os.environ['J_NOTEBOOK'] == '1':
    from preprocess.preprocess import Preprocess
else:
    from preprocess import Preprocess

from main import LGBMNumerical, lgb_f1_score


# def f1_score_w(y_true, y_pred, w):
#     actual_positives = y_true == 1
#     tp = np.sum((y_true[actual_positives] == y_pred[actual_positives]).astype(float)*w[actual_positives])
#     false_predictions = np.sum((y_true != y_pred).astype(float)*w)
#     return tp/(tp + .5*false_predictions)

# def load_images_blue():
#     paths = [f"{PATH}/{path}" for path in sorted(listdir(PATH))][:TRAIN_SIZE]
#     return [np.asarray(Image.open(path).convert('RGB'))[:, :, 2] for path in paths]

# IMAGES_B = load_images_blue()

# if PARAMETERS.DATASET == 'DRIVE':
#     with open(r'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/DRIVE/weights.pkl', mode='rb') as f:
#         W_TRAIN_FULL, W_TEST_FULL = pickle.load(f)
#         W_TRAIN_FULL[W_TRAIN_FULL == 0] = 0.3
#         W_TEST_FULL[W_TEST_FULL == 0] = 0.3
#     W_TRAIN = W_TRAIN_FULL[TRAIN_INDEX]
#     W_TEST = W_TEST_FULL[TEST_INDEX]
#
# def lgb_f1_score_w(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
#     return 'f1', f1_score_w(y_true, y_hat, W_TEST), True


# def f1(individual, n_kernels, k_size, *_, **__):
#     features = [pd.DataFrame()]*TRAIN_SIZE
#     count = 0
#     for j, ks in enumerate(k_size):
#         k_len = int(ks**2)
#         features = [
#             pd.concat([feat_df, pd.DataFrame(np.array(
#                 [cv2.filter2D(img, -1, individual[(count + i*k_len):(count + (i + 1) * k_len)].reshape((ks, ks)))[mask]
#                  for i in range(n_kernels // len(k_size))]
#             ).T, columns=np.arange(j * (n_kernels // len(k_size)), (j + 1) * (n_kernels // len(k_size))))], axis=1)
#             for img, mask, feat_df in zip(IMAGES, MASKS, features)
#         ]
#         count += k_len * (n_kernels // len(k_size))
#     CLF.fit(pd.concat(features[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :], Y_TRAIN, eval_metric=lgb_f1_score)
#     # CLF.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
#     y_pred = CLF.predict(pd.concat(features[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :])
#     return f1_score(Y_TEST, y_pred)
#
#
# Y_IMG = []
# IDX_IMG = []
# for j in range(TRAIN_SIZE):
#     y = pd.DataFrame(np.array(LABELS[j], dtype=int)[MASKS[j]].ravel())
#     idx = train_test_split(np.arange(y.shape[0]), test_size=0.2, random_state=42, stratify=y)[0]
#     Y_IMG.append(y.iloc[idx, :])
#     IDX_IMG.append(idx)
#
# Y_CV = []
# for j in range(TRAIN_SIZE):
#     Y_CV.append(pd.concat([item for l, item in enumerate(Y_IMG) if l != j], ignore_index=True))
#
# if False:
#     Y_TRAIN_FOLD = pd.DataFrame(
#         np.concatenate([np.array(label, dtype=int)[MASKS_FOLD[i]].ravel() for i, label in enumerate(LABELS_FOLD) if i < round(FOLD_SIZE*0.7)],
#                        axis=0).T).values.ravel()
#     Y_TEST_FOLD = pd.DataFrame(
#         np.concatenate([np.array(label, dtype=int)[MASKS_FOLD[i]].ravel() for i, label in enumerate(LABELS_FOLD) if i >= round(FOLD_SIZE*0.7)],
#                        axis=0).T).values.ravel()
#
#
#     TRAIN_INDEX_FOLD, _ = train_test_split(np.arange(Y_TRAIN_FOLD.shape[0]), test_size=0.2, random_state=42, stratify=Y_TRAIN_FOLD)
#     TEST_INDEX_FOLD, _ = train_test_split(np.arange(Y_TEST_FOLD.shape[0]), test_size=0.2, random_state=42, stratify=Y_TEST_FOLD)
#     Y_TRAIN_SAMPLE_FOLD = Y_TRAIN_FOLD[TRAIN_INDEX_FOLD]
#     Y_TEST_SAMPLE_FOLD = Y_TEST_FOLD[TEST_INDEX_FOLD]
#
#
# def f1_fold(individual, n_kernels, k_size, *_, **__):
#     features = [pd.DataFrame()]*FOLD_SIZE
#     count = 0
#     for j, ks in enumerate(k_size):
#         k_len = int(ks**2)
#         features = [
#             pd.concat([feat_df, pd.DataFrame(np.array(
#                 [cv2.filter2D(img, -1, individual[(count + i*k_len):(count + (i + 1) * k_len)].reshape((ks, ks)))[mask]
#                  for i in range(n_kernels // len(k_size))]
#             ).T, columns=np.arange(j * (n_kernels // len(k_size)), (j + 1) * (n_kernels // len(k_size))))], axis=1)
#             for img, mask, feat_df in zip(IMAGES_FOLD, MASKS_FOLD, features)
#         ]
#         count += k_len * (n_kernels // len(k_size))
#     CLF.fit(
#         pd.concat(features[:round(FOLD_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX_FOLD, :],
#         Y_TRAIN_SAMPLE_FOLD,
#         eval_metric=lgb_f1_score
#     )
#     y_pred = CLF.predict(pd.concat(features[round(FOLD_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX_FOLD, :])
#     return f1_score(Y_TEST_SAMPLE_FOLD, y_pred)
#
#
# def f1_preprocess(individual, *_, **__):
#     # clf = MultinomialNB(fit_prior=True)
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     # print(individual)
#     features = [
#         pd.DataFrame(P_OBJ.img_processing(np.asarray(img), params=individual)[mask])
#         for img, mask in zip(IMAGES, MASKS)
#     ]
#     CLF.fit(pd.concat(features[:round(TRAIN_SIZE*0.7)], ignore_index=True), Y_TRAIN_FULL, eval_metric=lgb_f1_score)
#     # clf.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
#     y_pred = CLF.predict(pd.concat(features[round(TRAIN_SIZE*0.7):], ignore_index=True))
#     # y_pred = clf.predict(pd.concat(features[10:], ignore_index=True))
#     return f1_score(Y_TEST_FULL, y_pred)
#
#
# def f1_preprocess_gb(individual, *_, **__):
#     # clf = MultinomialNB(fit_prior=True)
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     # print(individual)
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES]
#     preprocessed_images_b = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES_B]
#     green_p = [pd.DataFrame(img[mask], columns=['original_green']) for img, mask in zip(preprocessed_images, MASKS)]
#     blue_p = [pd.DataFrame(img[mask], columns=['original_blue']) for img, mask in zip(preprocessed_images_b, MASKS)]
#     df_train = pd.concat([
#         pd.concat(green_p[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(blue_p[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :]
#     ], axis=1)
#     CLF.fit(df_train, Y_TRAIN, eval_metric=lgb_f1_score)
#     df_test = pd.concat([
#         pd.concat(green_p[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(blue_p[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :]
#     ], axis=1)
#     y_pred = CLF.predict(df_test)
#     return f1_score(Y_TEST, y_pred)
#
#
# def f1_preprocess_w(individual, *_, **__):
#     # clf = MultinomialNB(fit_prior=True)
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     # print(individual)
#     features = [
#         pd.DataFrame(P_OBJ.img_processing(np.asarray(img), params=individual)[mask])
#         for img, mask in zip(IMAGES, MASKS)
#     ]
#     df_train = pd.concat([X_TRAIN_P, pd.concat(features[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :]], axis=1)
#     # CLF.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN_FULL, eval_metric=lgb_f1_score)
#     CLF.fit(df_train, Y_TRAIN, sample_weight=W_TRAIN, eval_metric=lgb_f1_score_w)
#     # clf.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
#     df_test = pd.concat([X_TEST_P, pd.concat(features[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :]], axis=1)
#     y_pred = CLF.predict(df_test)
#     # y_pred = clf.predict(pd.concat(features[10:], ignore_index=True))
#     return f1_score_w(Y_TEST, y_pred, W_TEST)
#
#
# def f1_preprocess_lbp(individual, *_, **__):
#     # clf = MultinomialNB(fit_prior=True)
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES]
#
#     images = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images]
#     i = 2
#     images = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images
#     ]
#     riu_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu'])
#         for img, mask in zip(images, MASKS_B)]
#     var_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var'])
#         for img, mask in zip(images, MASKS_B)]
#
#     preprocessed_images = [pd.DataFrame(img[mask], columns=['original']) for img, mask in zip(IMAGES, MASKS)]
#
#     df_train = pd.concat([
#         pd.concat(preprocessed_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(riu_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(var_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#     ], axis=1)
#     # CLF.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN_FULL, eval_metric=lgb_f1_score)
#     CLF.fit(df_train, Y_TRAIN, eval_metric=lgb_f1_score)
#     # clf.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
#     df_test = pd.concat([
#         pd.concat(preprocessed_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(riu_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(var_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#     ], axis=1)
#     y_pred = CLF.predict(df_test)
#     # y_pred = clf.predict(pd.concat(features[10:], ignore_index=True))
#     return f1_score(Y_TEST, y_pred)
#
#
# def f1_preprocess_lbp_gb(individual, *_, **__):
#     # clf = MultinomialNB(fit_prior=True)
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES]
#     preprocessed_images_b = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES_B]
#
#     images = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images]
#     images_b = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images_b]
#     i = 2
#     images_scaled = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images
#     ]
#     images_scaled_b = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images_b
#     ]
#     riu_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#     var_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#     riu_images_b = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu_b'])
#         for img, mask in zip(images_scaled_b, MASKS_B)]
#     var_images_b = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var_b'])
#         for img, mask in zip(images_scaled_b, MASKS_B)]
#
#     green_p = [pd.DataFrame(img[mask], columns=['original_green']) for img, mask in zip(preprocessed_images, MASKS)]
#     blue_p = [pd.DataFrame(img[mask], columns=['original_blue']) for img, mask in zip(preprocessed_images_b, MASKS)]
#
#     df_train = pd.concat([
#         pd.concat(green_p[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(blue_p[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(riu_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(var_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(riu_images_b[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(var_images_b[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#     ], axis=1)
#     # CLF.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN_FULL, eval_metric=lgb_f1_score)
#     CLF.fit(df_train, Y_TRAIN, eval_metric=lgb_f1_score)
#     # clf.fit(pd.concat(features[:10], ignore_index=True), Y_TRAIN)
#     df_test = pd.concat([
#         pd.concat(green_p[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(blue_p[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(riu_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(var_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(riu_images_b[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(var_images_b[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#     ], axis=1)
#     y_pred = CLF.predict(df_test)
#     # y_pred = clf.predict(pd.concat(features[10:], ignore_index=True))
#     return f1_score(Y_TEST, y_pred)
#
#
# def f1_preprocess_lbp_g_fold(individual, *_, **__):
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES_FOLD]
#     images = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images]
#     i = 2
#     images_scaled = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images
#     ]
#     riu_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu_g'])
#         for img, mask in zip(images_scaled, MASKS_RESCALED_FOLD)]
#     var_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var_g'])
#         for img, mask in zip(images_scaled, MASKS_RESCALED_FOLD)]
#
#     green_p = [pd.DataFrame(img[mask], columns=['original_green']) for img, mask in zip(preprocessed_images, MASKS_FOLD)]
#
#     df_train = pd.concat([
#         pd.concat(green_p[:round(FOLD_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX_FOLD, :],
#         pd.concat(riu_images[:round(FOLD_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX_FOLD, :],
#         pd.concat(var_images[:round(FOLD_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX_FOLD, :],
#     ], axis=1)
#     CLF.fit(df_train, Y_TRAIN_SAMPLE_FOLD, eval_metric=lgb_f1_score)
#     df_test = pd.concat([
#         pd.concat(green_p[round(FOLD_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX_FOLD, :],
#         pd.concat(riu_images[round(FOLD_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX_FOLD, :],
#         pd.concat(var_images[round(FOLD_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX_FOLD, :],
#     ], axis=1)
#     y_pred = CLF.predict(df_test)
#     return f1_score(Y_TEST_SAMPLE_FOLD, y_pred)
#
#
# def f1_preprocess_lbp_g_cv(individual, *_, **__):
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES]
#     images = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images]
#     i = 2
#     images_scaled = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images
#     ]
#     riu_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#     var_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#
#
#     green_p = [pd.DataFrame(img[mask], columns=['original_green']) for img, mask in zip(preprocessed_images, MASKS)]
#
#     f1_score_list = []
#     for j in range(TRAIN_SIZE):
#         df_train = pd.concat([
#             pd.concat([item.iloc[IDX_IMG[l], :] for l, item in enumerate(green_p) if l != j], ignore_index=True),
#             pd.concat([item.iloc[IDX_IMG[l], :] for l, item in enumerate(riu_images) if l != j], ignore_index=True),
#             pd.concat([item.iloc[IDX_IMG[l], :] for l, item in enumerate(var_images) if l != j], ignore_index=True),
#         ], axis=1)
#         CLF.fit(df_train, Y_CV[j].values.ravel(), eval_metric=lgb_f1_score)
#         df_test = pd.concat([
#             green_p[j].iloc[IDX_IMG[j], :],
#             riu_images[j].iloc[IDX_IMG[j], :],
#             var_images[j].iloc[IDX_IMG[j], :],
#         ], axis=1)
#         y_pred = CLF.predict(df_test)
#         f1_score_list.append(f1_score(Y_IMG[j].values.ravel(), y_pred))
#
#     return (min(f1_score_list) + sum(f1_score_list)/len(f1_score_list))/2


class Fitness:
    def __int__(self, fold=None):
        self.p_obj = Preprocess(
            lbp_radius=1,
            lbp_method=PARAMETERS.LBP_METHOD,
            height={'DRIVE': 608, 'CHASE': 960, 'STARE': 608}[PARAMETERS.DATASET],
            width={'DRIVE': 576, 'CHASE': 1024, 'STARE': 704}[PARAMETERS.DATASET],
            balance=PARAMETERS.BALANCE,
            fold=fold
        )
        self.clf = LGBMNumerical(
            num_leaves=50,
            max_depth=30,
            random_state=42,
            verbose=0,
            # metric='None',
            n_jobs=8,
            # n_estimators=1000,
            colsample_bytree=0.9,
            subsample=0.7,
            learning_rate=0.5,
            force_row_wise=True
        )
        self.path = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/images'
        self.masks_path = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/mask'
        self.labels_path = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/1st_manual'

        self.dataset_size = {'DRIVE': 20, 'DRIVE_TEST': 20, 'CHASE': 28, 'STARE': 20}[PARAMETERS.DATASET]
        self.train_size = 20 if PARAMETERS.DATASET == 'CHASE' else 14
        self.fold = fold
        if fold is not None:
            self.train_size = self.dataset_size - self.p_obj.fold_size

        self.images = self.load_images()
        self.masks = self.load_masks()
        self.rescaled_masks = [self.p_obj.rescale_add_borders(mask).astype(bool) for mask in self.masks]
        self.labels = self.load_labels()

        self.y_train = pd.DataFrame(np.concatenate(
            [np.array(label, dtype=int)[self.masks[i]].ravel() for i, label in enumerate(self.labels)
             if i < round(self.train_size * 0.7)],
            axis=0
        ).T).values.ravel()
        self.y_test = pd.DataFrame(np.concatenate(
            [np.array(label, dtype=int)[self.masks[i]].ravel() for i, label in enumerate(self.labels)
             if i >= round(self.train_size * 0.7)],
            axis=0
        ).T).values.ravel()
        self.train_index = train_test_split(
            np.arange(self.y_train.shape[0]), test_size=0.2, random_state=42, stratify=self.y_train)[0]
        self.test_index = train_test_split(
            np.arange(self.y_test.shape[0]), test_size=0.2, random_state=42, stratify=self.y_test)[0]
        self.sampled_y_train = self.y_train[self.train_index]
        self.sampled_y_test = self.y_test[self.test_index]

    def load_images(self):
        path_list = sorted(listdir(self.path))
        paths = [path_list[i_position] for i_position in self.p_obj.img_order][:self.train_size]
        return [np.asarray(Image.open(path).convert('RGB'))[:, :, 1] for path in paths]

    def load_masks(self):
        path_list = sorted(listdir(self.masks_path))
        paths = [path_list[i_position] for i_position in self.p_obj.img_order][:self.train_size]
        return [np.asarray(Image.open(path).convert('L')) > 100 for path in paths]

    def load_labels(self):
        path_list = sorted(listdir(self.labels_path))
        paths = [path_list[i_position] for i_position in self.p_obj.img_order][:self.train_size]
        return [np.asarray(Image.open(path).convert('L')) > 30 for path in paths]

    def f1_preprocess_lbp_g(self, individual, *_, **__):
        individual = np.round(individual).astype(int)
        individual[4] = max(1, individual[4])
        preprocessed_images = [self.p_obj.img_processing(np.asarray(img), params=individual) for img in self.images]
        images = [self.p_obj.rescale_add_borders(img) for img in preprocessed_images]
        i = 2
        images_scaled = [
            self.p_obj.rescale(img, (self.p_obj.width // i, self.p_obj.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
            for img in images
        ]
        riu_images = [
            pd.DataFrame(
                self.p_obj.repeat_pixels(self.p_obj.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask],
                columns=['riu_g']
            )
            for img, mask in zip(images_scaled, self.rescaled_masks)
        ]
        var_images = [
            pd.DataFrame(
                self.p_obj.repeat_pixels(self.p_obj.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask],
                columns=['var_g']
            )
            for img, mask in zip(images_scaled, self.rescaled_masks)
        ]

        green_p = [pd.DataFrame(img[mask], columns=['original_green'])
                   for img, mask in zip(preprocessed_images, self.masks)]

        df_train = pd.concat([
            pd.concat(green_p[:round(self.train_index*0.7)], ignore_index=True).iloc[self.train_index, :],
            pd.concat(riu_images[:round(self.train_index*0.7)], ignore_index=True).iloc[self.train_index, :],
            pd.concat(var_images[:round(self.train_index*0.7)], ignore_index=True).iloc[self.train_index, :],
        ], axis=1)
        self.clf.fit(df_train, self.sampled_y_train, eval_metric=lgb_f1_score)
        df_test = pd.concat([
            pd.concat(green_p[round(self.train_index*0.7):], ignore_index=True).iloc[self.test_index, :],
            pd.concat(riu_images[round(self.train_index*0.7):], ignore_index=True).iloc[self.test_index, :],
            pd.concat(var_images[round(self.train_index*0.7):], ignore_index=True).iloc[self.test_index, :],
        ], axis=1)
        y_predicted = self.clf.predict(df_test)
        return f1_score(self.sampled_y_test, y_predicted)

    def fitness_function(self, *args, **kwargs):
        return {
            'SPHERE': self.sphere,
            'RASTRIGIN': self.rastrigin,
            # 'F1': f1,
            # 'F1_FOLD': f1_fold,
            # 'PREPROCESS': f1_preprocess,
            # 'PREPROCESS_GB': f1_preprocess_gb,
            # 'PREPROCESS_W': f1_preprocess_w,
            # 'PREPROCESS_LBP': f1_preprocess_lbp,
            # 'PREPROCESS_LBP_GB': f1_preprocess_lbp_gb,
            'PREPROCESS_LBP_G': self.f1_preprocess_lbp_g,
            # 'PREPROCESS_LBP_G_FOLD': f1_preprocess_lbp_g_fold,
            # 'PREPROCESS_LBP_G_CV': f1_preprocess_lbp_g_cv,
        }[kwargs['function_name']](*args, **kwargs)

    @staticmethod
    def sphere(x, *_, **__):
        return np.sum(x**2)

    @staticmethod
    def rastrigin(x, n_kernels, *_, **__):
        return 10*n_kernels + np.sum(x**2 - 10*np.cos(2*np.pi*x))

# P_OBJ = Preprocess(
#     lbp_radius=1,
#     lbp_method=PARAMETERS.LBP_METHOD,
#     height={'DRIVE': 608, 'CHASE': 960, 'STARE': 608}[PARAMETERS.DATASET],
#     width={'DRIVE': 576, 'CHASE': 1024, 'STARE': 704}[PARAMETERS.DATASET],
#     balance=PARAMETERS.BALANCE
# )
#
# CLF = LGBMNumerical(
#     num_leaves=50,
#     max_depth=30,
#     random_state=42,
#     verbose=0,
#     # metric='None',
#     n_jobs=8,
#     # n_estimators=1000,
#     colsample_bytree=0.9,
#     subsample=0.7,
#     learning_rate=0.5,
#     force_row_wise=True
# )


# PATH = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/images'
# MASK_PATH = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/mask'
# LABELS_PATH = rf'/home/fer/Drive/Estudios/Master-IA/TFM/dataset/{PARAMETERS.DATASET}/training/1st_manual'
#
# TRAIN_SIZE = 20 if PARAMETERS.DATASET == 'CHASE' else 14


# def load_images():
#     paths = [f"{PATH}/{path}" for path in sorted(listdir(PATH))][:TRAIN_SIZE]
#     return [np.asarray(Image.open(path).convert('RGB'))[:, :, 1] for path in paths]
#
#
# def load_images_fold():
#     paths = [f"{PATH}/{sorted(listdir(PATH))[i_fold]}" for i_fold in FOLDS][:FOLD_SIZE]
#     return [np.asarray(Image.open(path).convert('RGB'))[:, :, 1] for path in paths]
#
#
# def load_images_blue():
#     paths = [f"{PATH}/{path}" for path in sorted(listdir(PATH))][:TRAIN_SIZE]
#     return [np.asarray(Image.open(path).convert('RGB'))[:, :, 2] for path in paths]
#
#
# def load_masks():
#     paths = [f"{MASK_PATH}/{path}" for path in sorted(listdir(MASK_PATH))][:TRAIN_SIZE]
#     return [np.asarray(Image.open(path).convert('L')) > 100 for path in paths]
#
#
# def load_masks_fold():
#     paths = [f"{MASK_PATH}/{sorted(listdir(MASK_PATH))[i_fold]}" for i_fold in FOLDS][:FOLD_SIZE]
#     return [np.asarray(Image.open(path).convert('L')) > 100 for path in paths]
#
#
# def load_labels():
#     paths = [f"{LABELS_PATH}/{path}" for path in sorted(listdir(LABELS_PATH))][:TRAIN_SIZE]
#     return [np.asarray(Image.open(path).convert('L')) > 30 for path in paths]
#
#
# def load_labels_fold():
#     paths = [f"{LABELS_PATH}/{sorted(listdir(LABELS_PATH))[i_fold]}" for i_fold in FOLDS][:FOLD_SIZE]
#     return [np.asarray(Image.open(path).convert('L')) > 30 for path in paths]

# IMAGES = load_images()
# MASKS = load_masks()
# LABELS = load_labels()

# MASKS_RESCALED_FOLD = [P_OBJ.rescale_add_borders(mask).astype(bool) for mask in MASKS_FOLD]

# Y_TRAIN_FULL = pd.DataFrame(
#     np.concatenate([np.array(label, dtype=int)[MASKS[i]].ravel() for i, label in enumerate(LABELS) if i < round(TRAIN_SIZE * 0.7)],
#                    axis=0).T).values.ravel()
# Y_TEST_FULL = pd.DataFrame(
#     np.concatenate([np.array(label, dtype=int)[MASKS[i]].ravel() for i, label in enumerate(LABELS) if i >= round(TRAIN_SIZE * 0.7)],
#                    axis=0).T).values.ravel()

# TRAIN_INDEX, _ = train_test_split(np.arange(Y_TRAIN_FULL.shape[0]), test_size=0.2, random_state=42, stratify=Y_TRAIN_FULL)
# TEST_INDEX, _ = train_test_split(np.arange(Y_TEST_FULL.shape[0]), test_size=0.2, random_state=42, stratify=Y_TEST_FULL)
# Y_TRAIN = Y_TRAIN_FULL[TRAIN_INDEX]
# Y_TEST = Y_TEST_FULL[TEST_INDEX]

# def f1_preprocess_lbp_g(individual, *_, **__):
#     individual = np.round(individual).astype(int)
#     individual[4] = max(1, individual[4])
#     preprocessed_images = [P_OBJ.img_processing(np.asarray(img), params=individual) for img in IMAGES]
#     images = [P_OBJ.rescale_add_borders(img) for img in preprocessed_images]
#     i = 2
#     images_scaled = [
#         P_OBJ.rescale(img, (P_OBJ.width // i, P_OBJ.height // i), algorithm=PARAMETERS.INTERPOLATION_ALGORITHM)
#         for img in images
#     ]
#     riu_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='riu', plot=PARAMETERS.PLOT), i)[mask], columns=['riu_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#     var_images = [
#         pd.DataFrame(
#             P_OBJ.repeat_pixels(P_OBJ.apply_lbp(img, method='var', plot=PARAMETERS.PLOT), i)[mask], columns=['var_g'])
#         for img, mask in zip(images_scaled, MASKS_B)]
#
#     green_p = [pd.DataFrame(img[mask], columns=['original_green']) for img, mask in zip(preprocessed_images, MASKS)]
#
#     df_train = pd.concat([
#         pd.concat(green_p[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(riu_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#         pd.concat(var_images[:round(TRAIN_SIZE*0.7)], ignore_index=True).iloc[TRAIN_INDEX, :],
#     ], axis=1)
#     CLF.fit(df_train, Y_TRAIN, eval_metric=lgb_f1_score)
#     df_test = pd.concat([
#         pd.concat(green_p[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(riu_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#         pd.concat(var_images[round(TRAIN_SIZE*0.7):], ignore_index=True).iloc[TEST_INDEX, :],
#     ], axis=1)
#     y_pred = CLF.predict(df_test)
#     return f1_score(Y_TEST, y_pred)
