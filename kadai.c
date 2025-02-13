/*
 * kadai.c
 *
 * 疎行列の2乗計算プログラム
 * 3種類のアルゴリズム（Dense, ハッシュベースSpGEMM, GustavsonベースSpGEMM）を採用
 * Gustavsonベースのアルゴリズムは、CSR, COO, List, Skyline 各形式に対して実装。
 *
 * 出力は各行「col value -1」形式
 * オプション -p 指定時は先頭に "rows cols" を出力
 * オプション -v 指定時は実行時間を stderr に出力
 *
 * コンパイル例:
 *   gcc kadai.c -O3 -std=c99 -o a.out
 *
 * 実行例:
 *   ./a.out -d csr -a gustavson -p -v < bcspwr01.txt > bcspwr01-2.txt
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #include "uthash.h"  // uthash ライブラリを同梱または適切なパスに配置
 
 /* =====================
    データ構造の定義
    ===================== */
 
 /* (1) Dense Matrix */
 typedef struct {
     int rows;
     int cols;
     double *data;  // row-major order: data[i*cols + j]
 } DenseMatrix;
 
 /* (2) CSR */
 typedef struct {
     int rows;
     int cols;
     int nnz;
     int *row_ptr;   // サイズ: rows+1
     int *col_idx;   // サイズ: nnz
     double *values; // サイズ: nnz
 } CSRMatrix;
 
 /* (3) COO */
 typedef struct {
     int row;
     int col;
     double value;
 } Triple;
 
 typedef struct {
     int rows;
     int cols;
     int nnz;
     Triple *triples;  // 動的配列
 } COOMatrix;
 
 /* (4) List of Lists */
 typedef struct ListNode {
     int col;
     double value;
     struct ListNode *next;
 } ListNode;
 
 typedef struct {
     int rows;
     int cols;
     ListNode **row_heads;  // サイズ: rows
 } ListMatrix;
 
 /* (5) Skyline Storage (下三角部分のみ保存、対称行列専用) */
 typedef struct {
     int n;  // 正方行列のサイズ
     int *profile;   // 各行の最小列番号（0-indexed）
     int *row_start; // 各行の開始位置（data 配列内のインデックス）、サイズ n+1
     double *data;   // 各行 i に対し、profile[i]～i の値を連続に格納
 } SkylineMatrix;
 
 /* =====================
    uthash 用の構造体（ハッシュベースSpGEMM用）
    ===================== */
 typedef struct {
     int col;         // キー: 列番号
     double value;    // 累積値
     UT_hash_handle hh;
 } HashEntry;
 
 void add_to_hash(HashEntry **table, int col, double val) {
     HashEntry *entry = NULL;
     HASH_FIND_INT(*table, &col, entry);
     if (entry == NULL) {
         entry = (HashEntry*)malloc(sizeof(HashEntry));
         entry->col = col;
         entry->value = val;
         HASH_ADD_INT(*table, col, entry);
     } else {
         entry->value += val;
     }
 }
 
 /* =====================
    入力関数
    入力形式:
      最初の行: "rows cols"
      各行: 1-indexed の "col value" ペアを連続し、行末に -1 が付く
    ===================== */
 static int read_int() {
     int x;
     if (scanf("%d", &x) != 1) { fprintf(stderr, "Error reading int.\n"); exit(EXIT_FAILURE); }
     return x;
 }
 static double read_double() {
     double x;
     if (scanf("%lf", &x) != 1) { fprintf(stderr, "Error reading double.\n"); exit(EXIT_FAILURE); }
     return x;
 }
 
 /* Dense Matrix の読み込み */
 DenseMatrix* read_dense() {
     int rows, cols;
     if (scanf("%d %d", &rows, &cols) != 2) { fprintf(stderr, "Failed to read matrix dimensions.\n"); exit(EXIT_FAILURE); }
     DenseMatrix *M = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     M->rows = rows; M->cols = cols;
     M->data = (double*)calloc(rows * cols, sizeof(double));
     for (int i = 0; i < rows; i++) {
         while (1) {
             int col = read_int();
             if (col == -1) break;
             double val = read_double();
             M->data[i * cols + (col - 1)] = val;
         }
     }
     return M;
 }
 
 /* CSR の読み込み */
 CSRMatrix* read_csr() {
     int rows, cols;
     if (scanf("%d %d", &rows, &cols) != 2) { fprintf(stderr, "Failed to read dimensions for CSR.\n"); exit(EXIT_FAILURE); }
     int *nnz_per_row = (int*)calloc(rows, sizeof(int));
     Triple **temp = (Triple**)malloc(rows * sizeof(Triple*));
     int *temp_count = (int*)calloc(rows, sizeof(int));
     int *temp_capacity = (int*)malloc(rows * sizeof(int));
     for (int i = 0; i < rows; i++) {
         temp_capacity[i] = 16;
         temp[i] = (Triple*)malloc(temp_capacity[i] * sizeof(Triple));
         temp_count[i] = 0;
     }
     for (int i = 0; i < rows; i++) {
         while (1) {
             int col = read_int();
             if (col == -1) break;
             double val = read_double();
             if (temp_count[i] >= temp_capacity[i]) {
                 temp_capacity[i] *= 2;
                 temp[i] = (Triple*)realloc(temp[i], temp_capacity[i] * sizeof(Triple));
             }
             temp[i][temp_count[i]].row = i;
             temp[i][temp_count[i]].col = col - 1;
             temp[i][temp_count[i]].value = val;
             temp_count[i]++;
             nnz_per_row[i]++;
         }
     }
     int total_nnz = 0;
     for (int i = 0; i < rows; i++) total_nnz += nnz_per_row[i];
     CSRMatrix *csr = (CSRMatrix*)malloc(sizeof(CSRMatrix));
     csr->rows = rows; csr->cols = cols; csr->nnz = total_nnz;
     csr->row_ptr = (int*)malloc((rows+1)*sizeof(int));
     csr->col_idx = (int*)malloc(total_nnz * sizeof(int));
     csr->values  = (double*)malloc(total_nnz * sizeof(double));
     csr->row_ptr[0] = 0;
     for (int i = 0; i < rows; i++) {
         csr->row_ptr[i+1] = csr->row_ptr[i] + nnz_per_row[i];
         int offset = csr->row_ptr[i];
         for (int j = 0; j < temp_count[i]; j++) {
             csr->col_idx[offset + j] = temp[i][j].col;
             csr->values[offset + j] = temp[i][j].value;
         }
         free(temp[i]);
     }
     free(temp); free(temp_count); free(temp_capacity); free(nnz_per_row);
     return csr;
 }
 
 /* COO の読み込み */
 COOMatrix* read_coo() {
     int rows, cols;
     if (scanf("%d %d", &rows, &cols) != 2) { fprintf(stderr, "Failed to read dimensions for COO.\n"); exit(EXIT_FAILURE); }
     int capacity = 1024;
     COOMatrix *coo = (COOMatrix*)malloc(sizeof(COOMatrix));
     coo->rows = rows; coo->cols = cols; coo->nnz = 0;
     coo->triples = (Triple*)malloc(capacity * sizeof(Triple));
     for (int i = 0; i < rows; i++) {
         while (1) {
             int col = read_int();
             if (col == -1) break;
             double val = read_double();
             if (coo->nnz >= capacity) {
                 capacity *= 2;
                 coo->triples = (Triple*)realloc(coo->triples, capacity * sizeof(Triple));
             }
             coo->triples[coo->nnz].row = i;
             coo->triples[coo->nnz].col = col - 1;
             coo->triples[coo->nnz].value = val;
             coo->nnz++;
         }
     }
     return coo;
 }
 
 /* List of Lists の読み込み */
 ListMatrix* read_list() {
     int rows, cols;
     if (scanf("%d %d", &rows, &cols) != 2) { fprintf(stderr, "Failed to read dimensions for List.\n"); exit(EXIT_FAILURE); }
     ListMatrix *L = (ListMatrix*)malloc(sizeof(ListMatrix));
     L->rows = rows; L->cols = cols;
     L->row_heads = (ListNode**)malloc(rows * sizeof(ListNode*));
     for (int i = 0; i < rows; i++) {
         L->row_heads[i] = NULL;
         while (1) {
             int col = read_int();
             if (col == -1) break;
             double val = read_double();
             ListNode *node = (ListNode*)malloc(sizeof(ListNode));
             node->col = col - 1;
             node->value = val;
             node->next = L->row_heads[i];
             L->row_heads[i] = node;
         }
     }
     return L;
 }
 
 /* Skyline の読み込み */
 SkylineMatrix* read_skyline() {
     int rows, cols;
     if (scanf("%d %d", &rows, &cols) != 2 || rows != cols) {
         fprintf(stderr, "Skyline requires a square matrix.\n");
         exit(EXIT_FAILURE);
     }
     int n = rows;
     double **rowvals = (double**)malloc(n * sizeof(double*));
     int *rowcap = (int*)malloc(n * sizeof(int));
     int *rowcount = (int*)calloc(n, sizeof(int));
     int *mincol = (int*)malloc(n * sizeof(int));
     for (int i = 0; i < n; i++) {
         rowcap[i] = 16;
         rowvals[i] = (double*)malloc(rowcap[i] * sizeof(double));
         rowcount[i] = 0;
         mincol[i] = i;  // 初期は対角のみ
     }
     for (int i = 0; i < n; i++) {
         while (1) {
             int col = read_int();
             if (col == -1) break;
             double val = read_double();
             int c = col - 1;
             if (c > i) continue;  // 上三角は無視（対称性により補完）
             if (c < mincol[i]) mincol[i] = c;
             if (rowcount[i] >= rowcap[i]) {
                 rowcap[i] *= 2;
                 rowvals[i] = (double*)realloc(rowvals[i], rowcap[i] * sizeof(double));
             }
             rowvals[i][rowcount[i]++] = val;
         }
     }
     SkylineMatrix *sky = (SkylineMatrix*)malloc(sizeof(SkylineMatrix));
     sky->n = n;
     sky->profile = (int*)malloc(n * sizeof(int));
     sky->row_start = (int*)malloc((n+1) * sizeof(int));
     sky->row_start[0] = 0;
     for (int i = 0; i < n; i++) {
         sky->profile[i] = mincol[i];
         int len = i - sky->profile[i] + 1;
         sky->row_start[i+1] = sky->row_start[i] + len;
     }
     int total = sky->row_start[n];
     sky->data = (double*)malloc(total * sizeof(double));
     for (int i = 0; i < n; i++) {
         int len = i - sky->profile[i] + 1;
         int start = sky->row_start[i];
         for (int j = 0; j < len; j++) {
             sky->data[start + j] = rowvals[i][j];
         }
         free(rowvals[i]);
     }
     free(rowvals); free(rowcap); free(rowcount); free(mincol);
     return sky;
 }
 
 /* =====================
    convert_to_dense() 関数群
    各疎行列を DenseMatrix に変換する
    ===================== */
 
 /* CSR → Dense */
 DenseMatrix* csr_convert_to_dense(CSRMatrix *A) {
     DenseMatrix *M = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     M->rows = A->rows; M->cols = A->cols;
     M->data = (double*)calloc(M->rows * M->cols, sizeof(double));
     for (int i = 0; i < A->rows; i++) {
         for (int idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; idx++) {
             int j = A->col_idx[idx];
             M->data[i * M->cols + j] = A->values[idx];
         }
     }
     return M;
 }
 
 /* COO → Dense */
 DenseMatrix* coo_convert_to_dense(COOMatrix *A) {
     DenseMatrix *M = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     M->rows = A->rows; M->cols = A->cols;
     M->data = (double*)calloc(M->rows * M->cols, sizeof(double));
     for (int i = 0; i < A->nnz; i++) {
         int r = A->triples[i].row;
         int c = A->triples[i].col;
         M->data[r * M->cols + c] = A->triples[i].value;
     }
     return M;
 }
 
 /* List → Dense */
 DenseMatrix* list_convert_to_dense(ListMatrix *L) {
     DenseMatrix *M = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     M->rows = L->rows; M->cols = L->cols;
     M->data = (double*)calloc(M->rows * M->cols, sizeof(double));
     for (int i = 0; i < L->rows; i++) {
         ListNode *node = L->row_heads[i];
         while (node) {
             M->data[i * M->cols + node->col] = node->value;
             node = node->next;
         }
     }
     return M;
 }
 
 /* Skyline → Dense */
 DenseMatrix* skyline_convert_to_dense(SkylineMatrix *sky) {
     int n = sky->n;
     DenseMatrix *M = (DenseMatrix*)malloc(sizeof(DenseMatrix));
     M->rows = n; M->cols = n;
     M->data = (double*)calloc(n * n, sizeof(double));
     for (int i = 0; i < n; i++) {
         int start = sky->row_start[i];
         int len = i - sky->profile[i] + 1;
         for (int j = 0; j < len; j++) {
             int col = sky->profile[i] + j;
             double val = sky->data[start + j];
             M->data[i * n + col] = val;
             if (i != col)
                 M->data[col * n + i] = val;
         }
     }
     return M;
 }
 
 /* =====================
    出力関数
    各行を "col value -1" 形式で出力
    オプション -p 指定時は先頭に "rows cols" を出力する
    ===================== */
 void output_result(int rows, int cols, double *res, int print_dims) {
     if (print_dims)
         printf("%d %d\n", rows, cols);
     for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
             double v = res[i * cols + j];
             if (v != 0.0)
                 printf("%d %.6lf ", j+1, v);
         }
         printf("-1\n");
     }
 }
 
 /* =====================
    乗算アルゴリズムの実装
    入力行列 A の2乗 (A^2) を Dense 結果（double*）として返す
    ===================== */
 
 /* --- Denseアルゴリズム --- */
 /* Dense Naïve: 単純な三重ループ */
 double* dense_naive(DenseMatrix *A) {
     int m = A->rows, n = A->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     for (int i = 0; i < m; i++)
         for (int j = 0; j < n; j++) {
             double sum = 0.0;
             for (int k = 0; k < n; k++)
                 sum += A->data[i*n + k] * A->data[k*n + j];
             res[i*n + j] = sum;
         }
     return res;
 }
 
 /* --- ハッシュベースSpGEMM --- */
 /* CSR SpGEMM */
 double* csr_hash(CSRMatrix *A) {
     int m = A->rows, n = A->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     for (int i = 0; i < m; i++) {
         HashEntry *row_hash = NULL;
         for (int idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; idx++) {
             int k = A->col_idx[idx];
             double a = A->values[idx];
             for (int jdx = A->row_ptr[k]; jdx < A->row_ptr[k+1]; jdx++) {
                 int j = A->col_idx[jdx];
                 double b = A->values[jdx];
                 add_to_hash(&row_hash, j, a * b);
             }
         }
         HashEntry *entry, *tmp;
         HASH_ITER(hh, row_hash, entry, tmp) {
             res[i*n + entry->col] = entry->value;
             HASH_DEL(row_hash, entry);
             free(entry);
         }
     }
     return res;
 }
 
 /* COO SpGEMM */
 double* coo_hash(COOMatrix *A) {
     int m = A->rows, n = A->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     for (int i = 0; i < m; i++) {
         HashEntry *row_hash = NULL;
         // Assuming input is provided per row in order.
         for (int t = 0; t < A->nnz; t++) {
             if (A->triples[t].row == i) {
                 int k = A->triples[t].col;
                 double a = A->triples[t].value;
                 for (int s = 0; s < A->nnz; s++) {
                     if (A->triples[s].row == k) {
                         int j = A->triples[s].col;
                         double b = A->triples[s].value;
                         add_to_hash(&row_hash, j, a * b);
                     }
                 }
             }
         }
         HashEntry *entry, *tmp;
         HASH_ITER(hh, row_hash, entry, tmp) {
             res[i*n + entry->col] = entry->value;
             HASH_DEL(row_hash, entry);
             free(entry);
         }
     }
     return res;
 }
 
 /* List SpGEMM */
 double* list_hash(ListMatrix *L) {
     int m = L->rows, n = L->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     for (int i = 0; i < m; i++) {
         HashEntry *row_hash = NULL;
         ListNode *node = L->row_heads[i];
         while (node) {
             int k = node->col;
             double a = node->value;
             ListNode *node2 = L->row_heads[k];
             while (node2) {
                 int j = node2->col;
                 double b = node2->value;
                 add_to_hash(&row_hash, j, a * b);
                 node2 = node2->next;
             }
             node = node->next;
         }
         HashEntry *entry, *tmp;
         HASH_ITER(hh, row_hash, entry, tmp) {
             res[i*n + entry->col] = entry->value;
             HASH_DEL(row_hash, entry);
             free(entry);
         }
     }
     return res;
 }
 
 /* Skyline SpGEMM */
 double* skyline_hash(SkylineMatrix *sky) {
     int n = sky->n;
     double *res = (double*)calloc(n * n, sizeof(double));
     for (int i = 0; i < n; i++) {
         HashEntry *row_hash = NULL;
         for (int k = 0; k < n; k++) {
             double a = 0.0;
             if (i >= k) {
                 int idx = sky->row_start[i] + (k - sky->profile[i]);
                 if (k >= sky->profile[i] && k <= i) a = sky->data[idx];
             } else {
                 int idx = sky->row_start[k] + (i - sky->profile[k]);
                 if (i >= sky->profile[k] && i <= k) a = sky->data[idx];
             }
             if (a == 0.0) continue;
             for (int j = i; j < n; j++) {
                 double b = 0.0;
                 if (j >= k) {
                     int idx = sky->row_start[j] + (k - sky->profile[j]);
                     if (k >= sky->profile[j] && k <= j) b = sky->data[idx];
                 } else {
                     int idx = sky->row_start[k] + (j - sky->profile[k]);
                     if (j >= sky->profile[k] && j <= k) b = sky->data[idx];
                 }
                 if (b == 0.0) continue;
                 add_to_hash(&row_hash, j, a * b);
             }
         }
         HashEntry *entry, *tmp;
         HASH_ITER(hh, row_hash, entry, tmp) {
             res[i*n + entry->col] = entry->value;
             if (i != entry->col)
                 res[entry->col*n + i] = entry->value;
             HASH_DEL(row_hash, entry);
             free(entry);
         }
     }
     return res;
 }
 
 /* --- GustavsonベースSpGEMM --- */
 /* 各形式で、ワークスペース配列を用いて各行の結果を累積する */
 
 /* CSR Gustavson */
 double* csr_gustavson(CSRMatrix *A) {
     int m = A->rows, n = A->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     double *workspace = (double*)calloc(n, sizeof(double));
     for (int i = 0; i < m; i++) {
         for (int idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; idx++) {
             int k = A->col_idx[idx];
             double a = A->values[idx];
             for (int jdx = A->row_ptr[k]; jdx < A->row_ptr[k+1]; jdx++) {
                 int j = A->col_idx[jdx];
                 workspace[j] += a * A->values[jdx];
             }
         }
         for (int j = 0; j < n; j++) {
             if (workspace[j] != 0.0) {
                 res[i*n + j] = workspace[j];
                 workspace[j] = 0.0;
             }
         }
     }
     free(workspace);
     return res;
 }
 
 /* COO Gustavson */
 /* 前提: COO入力は行順に与えられている */
 double* coo_gustavson(COOMatrix *A) {
     int m = A->rows, n = A->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     double *workspace = (double*)calloc(n, sizeof(double));
     // 事前に各行の開始・終了インデックスを求める
     int *row_start = (int*)calloc(m, sizeof(int));
     int *row_end = (int*)calloc(m, sizeof(int));
     int current = 0;
     for (int i = 0; i < m; i++) {
         row_start[i] = current;
         while (current < A->nnz && A->triples[current].row == i)
             current++;
         row_end[i] = current;
     }
     for (int i = 0; i < m; i++) {
         for (int t = row_start[i]; t < row_end[i]; t++) {
             int k = A->triples[t].col;
             double a = A->triples[t].value;
             for (int s = row_start[k]; s < row_end[k]; s++) {
                 int j = A->triples[s].col;
                 workspace[j] += a * A->triples[s].value;
             }
         }
         for (int j = 0; j < n; j++) {
             if (workspace[j] != 0.0) {
                 res[i*n + j] = workspace[j];
                 workspace[j] = 0.0;
             }
         }
     }
     free(workspace); free(row_start); free(row_end);
     return res;
 }
 
 /* List Gustavson */
 double* list_gustavson(ListMatrix *L) {
     int m = L->rows, n = L->cols;
     double *res = (double*)calloc(m * n, sizeof(double));
     double *workspace = (double*)calloc(n, sizeof(double));
     for (int i = 0; i < m; i++) {
         for (ListNode *node = L->row_heads[i]; node; node = node->next) {
             int k = node->col;
             double a = node->value;
             for (ListNode *node2 = L->row_heads[k]; node2; node2 = node2->next) {
                 int j = node2->col;
                 workspace[j] += a * node2->value;
             }
         }
         for (int j = 0; j < n; j++) {
             if (workspace[j] != 0.0) {
                 res[i*n + j] = workspace[j];
                 workspace[j] = 0.0;
             }
         }
     }
     free(workspace);
     return res;
 }
 
 /* Skyline Gustavson */
 double* skyline_gustavson(SkylineMatrix *sky) {
     int n = sky->n;
     double *res = (double*)calloc(n * n, sizeof(double));
     double *workspace = (double*)calloc(n, sizeof(double));
     for (int i = 0; i < n; i++) {
         int start = sky->row_start[i];
         int len = i - sky->profile[i] + 1;
         for (int offset = 0; offset < len; offset++) {
             int k = sky->profile[i] + offset;
             double a = sky->data[start + offset];
             int start_k = sky->row_start[k];
             int len_k = k - sky->profile[k] + 1;
             for (int offset2 = 0; offset2 < len_k; offset2++) {
                 int j = sky->profile[k] + offset2;
                 workspace[j] += a * sky->data[start_k + offset2];
             }
         }
         for (int j = 0; j < n; j++) {
             if (workspace[j] != 0.0) {
                 res[i*n + j] = workspace[j];
                 workspace[j] = 0.0;
             }
         }
     }
     free(workspace);
     return res;
 }
 
 /* =====================
    メモリ解放関数
    ===================== */
 void free_dense(DenseMatrix *M) {
     if (M) { free(M->data); free(M); }
 }
 void free_csr(CSRMatrix *A) {
     if (A) { free(A->row_ptr); free(A->col_idx); free(A->values); free(A); }
 }
 void free_coo(COOMatrix *A) {
     if (A) { free(A->triples); free(A); }
 }
 void free_list(ListMatrix *L) {
     if (L) {
         for (int i = 0; i < L->rows; i++) {
             ListNode *node = L->row_heads[i];
             while (node) {
                 ListNode *tmp = node;
                 node = node->next;
                 free(tmp);
             }
         }
         free(L->row_heads);
         free(L);
     }
 }
 void free_skyline(SkylineMatrix *sky) {
     if (sky) { free(sky->profile); free(sky->row_start); free(sky->data); free(sky); }
 }
 
 /* =====================
    main() 関数
    オプション:
      -d {dense|csr|coo|list|skyline}  : データ構造選択（デフォルト: dense）
      -a {dense|hash|gustavson}        : 乗算アルゴリズム選択（デフォルト: dense）
      -p                              : 結果出力時に先頭に "rows cols" を出力
      -v                              : verbose, 実行時間を stderr に出力
    ===================== */
 int main(int argc, char *argv[]) {
     char ds_type[16] = "dense";
     char alg_type[16] = "dense";
     int print_dims = 0;
     int verbose = 0;
     for (int i = 1; i < argc; i++) {
         if (strcmp(argv[i], "-d") == 0 && i+1 < argc) {
             strncpy(ds_type, argv[i+1], sizeof(ds_type)-1);
             i++;
         } else if (strcmp(argv[i], "-a") == 0 && i+1 < argc) {
             strncpy(alg_type, argv[i+1], sizeof(alg_type)-1);
             i++;
         } else if (strcmp(argv[i], "-p") == 0) {
             print_dims = 1;
         } else if (strcmp(argv[i], "-v") == 0) {
             verbose = 1;
         } else {
             fprintf(stderr, "Usage: %s -d {dense|csr|coo|list|skyline} -a {dense|hash|gustavson} [-p] [-v]\n", argv[0]);
             exit(EXIT_FAILURE);
         }
     }
 
     clock_t start = clock();
     double *result = NULL;
     int rows = 0, cols = 0;
     
     if (strcmp(ds_type, "dense") == 0) {
         DenseMatrix *A = read_dense();
         rows = A->rows; cols = A->cols;
         /* Denseの場合は dense_naive をそのまま利用 */
         result = dense_naive(A);
         free_dense(A);
     }
     else if (strcmp(ds_type, "csr") == 0) {
         CSRMatrix *A = read_csr();
         rows = A->rows; cols = A->cols;
         if (strcmp(alg_type, "dense") == 0) {
             DenseMatrix *D = csr_convert_to_dense(A);
             result = dense_naive(D);
             free(D->data); free(D);
         }
         else if (strcmp(alg_type, "hash") == 0)
             result = csr_hash(A);
         else if (strcmp(alg_type, "gustavson") == 0)
             result = csr_gustavson(A);
         else {
             fprintf(stderr, "Unknown algorithm type for CSR.\n");
             exit(EXIT_FAILURE);
         }
         free_csr(A);
     }
     else if (strcmp(ds_type, "coo") == 0) {
         COOMatrix *A = read_coo();
         rows = A->rows; cols = A->cols;
         if (strcmp(alg_type, "dense") == 0) {
             DenseMatrix *D = coo_convert_to_dense(A);
             result = dense_naive(D);
             free(D->data); free(D);
         }
         else if (strcmp(alg_type, "hash") == 0)
             result = coo_hash(A);
         else if (strcmp(alg_type, "gustavson") == 0)
             result = coo_gustavson(A);
         else {
             fprintf(stderr, "Unknown algorithm type for COO.\n");
             exit(EXIT_FAILURE);
         }
         free_coo(A);
     }
     else if (strcmp(ds_type, "list") == 0) {
         ListMatrix *A = read_list();
         rows = A->rows; cols = A->cols;
         if (strcmp(alg_type, "dense") == 0) {
             DenseMatrix *D = list_convert_to_dense(A);
             result = dense_naive(D);
             free(D->data); free(D);
         }
         else if (strcmp(alg_type, "hash") == 0)
             result = list_hash(A);
         else if (strcmp(alg_type, "gustavson") == 0)
             result = list_gustavson(A);
         else {
             fprintf(stderr, "Unknown algorithm type for List.\n");
             exit(EXIT_FAILURE);
         }
         free_list(A);
     }
     else if (strcmp(ds_type, "skyline") == 0) {
         SkylineMatrix *A = read_skyline();
         rows = A->n; cols = A->n;
         if (strcmp(alg_type, "dense") == 0) {
             DenseMatrix *D = skyline_convert_to_dense(A);
             result = dense_naive(D);
             free(D->data); free(D);
         }
         else if (strcmp(alg_type, "hash") == 0)
             result = skyline_hash(A);
         else if (strcmp(alg_type, "gustavson") == 0)
             result = skyline_gustavson(A);
         else {
             fprintf(stderr, "Unknown algorithm type for Skyline.\n");
             exit(EXIT_FAILURE);
         }
         free_skyline(A);
     }
     else {
         fprintf(stderr, "Unknown data structure type: %s\n", ds_type);
         exit(EXIT_FAILURE);
     }
     
     output_result(rows, cols, result, print_dims);
     free(result);
     
     clock_t end = clock();
     if (verbose) {
         double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
         fprintf(stderr, "Time elapsed: %.6f sec\n", elapsed);
     }
     return 0;
 }