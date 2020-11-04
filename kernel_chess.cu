#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define _USE_MATH_DEFINES
#define MAX_MOVES 32
#define MINIMAX_MAXVAL 50000
#define MINIMAX_MINVAL -50000
#define MASK 0xFFFFFFFF

using namespace std;
using namespace std::chrono;


//---------------------------------------------Funkcje GPU------------------------------------------------------------------
//-------------Szukanie pionkow gracza na planszy-----------------
__global__ void kernelFindPawns(int* board, int* pawns_id, bool whose_move)
{
	if(threadIdx.x < 64) {
		int pawn_pos = 1000;

		if(whose_move == 0 && board[threadIdx.x] > 0 && board[threadIdx.x] < 10)
			pawn_pos = threadIdx.x;
		else if(whose_move == 1 && board[threadIdx.x] > 10)
			pawn_pos = threadIdx.x;
		
		pawns_id[threadIdx.x] = pawn_pos;
	}
}

//-------------Wyznaczanie mozliwych ruchow-----------------
__inline__ __device__ 
bool checkMovePawn(int* board, int start_pos, int end_pos, bool whose_move)
{
	int end_field = board[end_pos];
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	bool move_possible = 0;
	
	if(whose_move == 0){
		if(start_col == end_col) {
			if(end_row == start_row - 1 && end_field == 0)
				 move_possible = 1;
			else if(end_row == start_row - 2 && end_field == 0 && start_row == 6) {
				if(board[(start_row - 1)*8 + end_col] == 0)
					move_possible = 1;
			}			 
		}
		else if(abs(start_col - end_col) == 1) {
			if(end_row == start_row - 1 && end_field != 0)
				 move_possible = 1;
		}
	}
	else{
		if(start_col == end_col) {
			if(end_row == start_row + 1 && end_field == 0)
				 move_possible = 1;
			else if(end_row == start_row + 2 && end_field == 0 && start_row == 1) {
				if(board[(start_row + 1)*8 + end_col] == 0)
					move_possible = 1;
			}			 
		}
		else if(abs(start_col - end_col) == 1) {
			if(end_row == start_row + 1 && end_field != 0)
				 move_possible = 1;
		}
	}

	return move_possible;
}

__inline__ __device__ 
bool checkMoveKnight(int* board, int start_pos, int end_pos)
{
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	bool move_possible = 0;
	
	if(abs(start_col - end_col) == 2 && abs(start_row - end_row) == 1)
		move_possible = 1;
	else if(abs(start_col - end_col) == 1 && abs(start_row - end_row) == 2)
		move_possible = 1;

	return move_possible;
}

__inline__ __device__ 
bool checkMoveRook(int* board, int start_pos, int end_pos)
{
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	bool move_possible = 0;
	
	if(end_col == start_col) {
		move_possible = 1;
		int delta_row = end_row - start_row;
		int row_mult = 1;
		if(delta_row < 0)
			row_mult = -1;
		else
			delta_row++;

		for(int i = 1; i < abs(delta_row); i++) {
			int id = (start_row + i * row_mult) * 8 + end_col;
			if(board[id] != 0) {
				move_possible = 0;
				return move_possible;
			}
		}
	}
	else if(end_row == start_row) {
		move_possible = 1;
		int delta_col = end_col - start_col;
		int col_mult = 1;
		if(delta_col < 0)
			col_mult = -1;
		else
			delta_col++;

		for(int i = 1; i < abs(delta_col); i++) {
			int id = start_row * 8 + start_col + i * col_mult;
			if(board[id] != 0) {
				move_possible = 0;
				return move_possible;
			}
		}
	}

	return move_possible;
}

__inline__ __device__ 
bool checkMoveBishop(int* board, int start_pos, int end_pos)
{
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	int delta_row = end_row - start_row;
	int delta_col = end_col - start_col;
	bool move_possible = 0;

	if(abs(delta_row) == abs(delta_col)) {
		move_possible = 1;
		int row_mult = 1;
		int col_mult = 1;
		if(delta_row < 0)
			row_mult = -1;
		if(delta_col < 0)
			col_mult = -1;

		for(int i = 1; i < abs(delta_row); i++) {
			int id = (start_row + i * row_mult) * 8 + start_col + i * col_mult;
			if(board[id] != 0) {
				move_possible = 0;
				return move_possible;
			}
		}
		
	}

	return move_possible;
}

__inline__ __device__ 
bool checkMoveQueen(int* board, int start_pos, int end_pos)
{
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	bool move_possible = 0;

	if(start_col == end_col || start_row == end_row)
		move_possible = checkMoveRook(board, start_pos, end_pos);
	else if(abs(start_row - end_row) == abs(start_col - end_col))
		move_possible = checkMoveBishop(board, start_pos, end_pos);

	return move_possible;
}

__inline__ __device__ 
bool checkMoveKing(int* board, int start_pos, int end_pos)
{
	int start_row = start_pos / 8;
	int start_col = start_pos % 8;
	int end_row = end_pos / 8;
	int end_col = end_pos % 8;
	bool move_possible = 0;
	
	if(abs(start_col - end_col) == 1 ||abs(start_col - end_col) == 0) {
		if(abs(start_row - end_row) == 1 || abs(start_row - end_row) == 0)
			move_possible = 1;
	}

	return move_possible;
}


__global__ void kernelCheckAllMoves(int* board, int* moves, int start_id, bool whose_move)
{
	int pawn = board[start_id] % 10;
	if(threadIdx.x < 64) {
		int end_id = threadIdx.x;
		bool move_possible;

		if(whose_move == 0 && board[end_id] > 0 && board[end_id] < 10)
			move_possible = 0;
		else if(whose_move == 1 && board[end_id] > 10)
			move_possible = 0;
		else {
			if(pawn == 1)
				move_possible = checkMovePawn(board, start_id, end_id, whose_move);
			else if(pawn == 2)
				move_possible = checkMoveRook(board, start_id, end_id);
			else if(pawn == 3)
				move_possible = checkMoveBishop(board, start_id, end_id);
			else if(pawn == 4)
				move_possible = checkMoveKnight(board, start_id, end_id);
			else if(pawn == 5)
				move_possible = checkMoveQueen(board, start_id, end_id);
			else if(pawn == 6)
				move_possible = checkMoveKing(board, start_id, end_id);
		}

		int ret_val = 1000;
		if(move_possible == 1)
			ret_val = threadIdx.x;
		moves[threadIdx.x] = ret_val;
	}
}

__global__ void kernelCheckMove(int* board, int* move, int start_id, int end_id, bool whose_move)
{
	int pawn = board[start_id] % 10;
	if(threadIdx.x == 0) {
		bool move_possible;

		if(whose_move == 0 && board[end_id] > 0 && board[end_id] < 10)
			move_possible = 0;
		else if(whose_move == 1 && board[end_id] > 10)
			move_possible = 0;
		else {
			if(pawn == 1)
				move_possible = checkMovePawn(board, start_id, end_id, whose_move);
			else if(pawn == 2)
				move_possible = checkMoveRook(board, start_id, end_id);
			else if(pawn == 3)
				move_possible = checkMoveBishop(board, start_id, end_id);
			else if(pawn == 4)
				move_possible = checkMoveKnight(board, start_id, end_id);
			else if(pawn == 5)
				move_possible = checkMoveQueen(board, start_id, end_id);
			else if(pawn == 6)
				move_possible = checkMoveKing(board, start_id, end_id);
		}
		int ret_val = 1000;
		if(move_possible == 1)
			ret_val = threadIdx.x;
		move[threadIdx.x] = ret_val;
	}
}


//--------------------Obliczanie punktow planszy-------------------
__inline__ __device__ 
int getPointsPawn(int pos, int field_type, bool whose_move)
{
	int row = pos / 8;
	int col = pos % 8;
	int points = 10;

	if(col == 0 || col == 7)
		pos -= 2;
	
	if(field_type < 10) {
		if(row == 1)
			points += 30;
		else if(row == 0)
			points += 70;
	}
	else {
		if(row == 6)
			points += 30;
		else if(row == 7)
			points += 70;
	}

	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}

__inline__ __device__ 
int getPointsBishop(int pos, int field_type, bool whose_move)
{
	int row = pos / 8;
	int col = pos % 8;
	int points = 30;

	if(row == 0 || row == 7)
		points -= 4;
	else if(row >= 2 && row <= 5)
		points += 4;

	if(col == 0 || col == 7)
		points -= 4;
	else if(col >= 2 && col <= 5)
		points += 4;
	
	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}

__inline__ __device__ 
int getPointsKnight(int pos, int field_type, bool whose_move)
{
	int row = pos / 8;
	int col = pos % 8;
	int points = 30;

	if(whose_move == 0 && row == 0)
		points -= 8;
	else if(whose_move == 0 && row == 7)
		points -= 8;
	else if(row == 0 || row == 7)
		points -= 4;
	else if(row >= 2 && row <= 5)
		points += 4;

	if(col == 0 || col == 7)
		points -= 4;
	else if(col >= 2 && col <= 5)
		points += 4;
	
	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}

__inline__ __device__ 
int getPointsRook(int pos, int field_type, bool whose_move)
{
	//int row = pos / 8;
	int col = pos % 8;
	int points = 50;

	if(col == 0 || col == 7)
		pos -= 6;
	else if(col == 1 || col == 6)
		pos -= 3;
	else if(col == 3 || col == 4)
		pos += 3;

	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}

__inline__ __device__ 
int getPointsQueen(int pos, int field_type, bool whose_move)
{
	//int row = pos / 8;
	int col = pos % 8;
	int points = 90;

	if(col == 0 || col == 7)
		points -= 5;
	
	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}

__inline__ __device__ 
int getPointsKing(int pos, int field_type, bool whose_move)
{
	int row = pos / 8;
	int col = pos % 8;
	int points = 1500;

	if(row > 1 && row < 6)
		points -= 20;
	else if(row == 1 || row == 6)
		points -= 5;

	if(col == 3 || col == 4)
		points -= 5;

	if(whose_move == 0 && field_type > 10)
		points *= -1;
	else if(whose_move == 1 && field_type < 10)
		points *= -1;

	return points;
}


__inline__ __device__
int warpReductionPoints(int value) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    value += __shfl_down_sync(MASK, value, offset);

  return value;
}

__inline__ __device__ 
int blockReductionPoints(int value)
{
	static __shared__ double shared[32];
	int w_id = threadIdx.x / warpSize;
	int t_id = threadIdx.x % warpSize;

	value = warpReductionPoints(value);

	if(t_id == 0)
		shared[w_id] = value;

	__syncthreads();

	value = (threadIdx.x < blockDim.x / warpSize) ? shared[t_id] : 0;

	//Finalna redukcja w pierwszym warpie
	if(w_id == 0)
		value = warpReductionPoints(value);

	return value;
}

__global__ void kernelEvalPoints(const int* board, int* points_out, bool whose_move)
{
	if(threadIdx.x < 64) {
		int pos_id = threadIdx.x;
		int field = board[pos_id];
		int points = 0;

		if(field == 0)
			points = 0;
		else if(field % 10 == 1) 
			points = getPointsPawn(pos_id, field, whose_move);
		else if(field % 10 == 2)
			points = getPointsRook(pos_id, field, whose_move);
		else if(field % 10 == 3) 
			points = getPointsBishop(pos_id, field, whose_move);
		else if(field % 10 == 4) 
			points = getPointsKnight(pos_id, field, whose_move);
		else if(field % 10 == 5) 
			points = getPointsQueen(pos_id, field, whose_move);
		else if(field % 10 == 6) 
			points = getPointsKing(pos_id, field, whose_move);

		points = blockReductionPoints(points);
		if(threadIdx.x == 0)
			points_out[blockIdx.x] = points;
	}
}

//----------------------------Minimax-------------------------------------------
__global__ void kernelMax(int* max, int* points, unsigned int rozmiar)
{
	__shared__ int max_values[MAX_MOVES];
    int t_id = threadIdx.x;
	unsigned int b_id = blockIdx.x * blockDim.x + threadIdx.x;

	max_values[t_id] = points[b_id];
	__syncthreads();

	for (int i = 1; i < MAX_MOVES; i *= 2) {
		if((t_id + i) < MAX_MOVES && max_values[t_id + i] != MINIMAX_MAXVAL) {
			if (max_values[t_id + i] > max_values[t_id])
				max_values[t_id] = max_values[t_id + i];
		}
		__syncthreads();
	}
			
	if (t_id == 0)
		max[blockIdx.x] = max_values[t_id];
}

__global__ void kernelMin(int* min, int* points, unsigned int rozmiar)
{
	__shared__ int min_values[MAX_MOVES];
    int t_id = threadIdx.x;
	unsigned int b_id = blockIdx.x * blockDim.x + threadIdx.x;

	min_values[t_id] = points[b_id];
	__syncthreads();

	for (int i = 1; i < MAX_MOVES; i *= 2) {
		if((t_id + i) < MAX_MOVES) {
			if (min_values[t_id + i] < min_values[t_id] && min_values[t_id + i] != MINIMAX_MINVAL)
				min_values[t_id] = min_values[t_id + i];
		}
		__syncthreads();
	}
	
	if (t_id == 0)
		min[blockIdx.x] = min_values[t_id];	
}


//------------------------------------------------Funkcje globalne----------------------------------------------------------
void resetuj_plansze(int** plansza, int* gracze) 
{
	/*Oznaczenia pionkow:
	0 - puste pole
	1/11 - pion gracza bialego/czarnego
	2/12 - wieza gracza bialego/czarnego
	3/13 - goniec gracza bialego/czarnego
	4/14 - skoczek gracza bialego/czarnego
	5/15 - hetman gracza bialego/czarnego
	6/16 - krol gracza bialego/czarnego
	*/
	for(int i = 0; i < 8; i++) {
		for(int j = 0; j < 8; j++) {
			int id = 0;
			//Dwa gorne rzedy na pionki gracza czarnego
			if(i == 0) {
				if(j == 0 || j == 7)
					id = 12;
				else if(j == 1 || j == 6)
					id = 14;
				else if(j == 2 || j == 5)
					id = 13;
				else if(j == 3)
					id = 15;
				else if(j == 4)
					id = 16;
				
			}
			else if(i == 1) {
				id = 11;
			}
			//Dwa dolne rzedy na pionki gracza bialego
			else if(i == 7) {
				if(j == 0 || j == 7)
					id = 2;
				else if(j == 1 || j == 6)
					id = 4;
				else if(j == 2 || j == 5)
					id = 3;
				else if(j == 3)
					id = 5;
				else if(j == 4)
					id = 6;
				
			}
			else if(i == 6) {
				id = 1;
			}
			//Reszta pol bez pionkow

			plansza[i][j] = id;
		}	
	}
	while(true) {
		system("clear");
		int wartosc = -1;
		cout << "Podaj gracza białego (0 - człowiek, 1 - AI)" << endl;
		cin >> wartosc;
		if(wartosc == 0 || wartosc == 1) {
			gracze[0] = wartosc;
			break;
		}
	}
	while(true) {
		system("clear");
		int wartosc = -1;
		cout << "Podaj gracza czarnego (0 - człowiek, 1 - AI)" << endl;
		cin >> wartosc;
		if(wartosc == 0 || wartosc == 1) {
			gracze[1] = wartosc;
			break;
		}
	}
}

void rysuj_plansze(int** plansza) 
{
	cout << "=||";
	for(int i = 0; i < 8; i++) {
		cout << "===|";
	}
	cout << "|";
	cout << endl;
	for(int i = 0; i < 8; i++) {
		cout << 8 - i;
		cout << "||";
		for(int j = 0; j < 8; j++) {
			int id = plansza[i][j];
			char rysuj = ' ';
			
			if(id % 10 == 1) {
				if(id < 10)
					rysuj = 'P';
				else
					rysuj = 'p';
			}
			else if(id % 10 == 2) {
				if(id < 10)
					rysuj = 'R';
				else
					rysuj = 'r';
			}
			else if(id % 10 == 3) {
				if(id < 10)
					rysuj = 'B';
				else
					rysuj = 'b';
			}
			else if(id % 10 == 4) {
				if(id < 10)
					rysuj = 'N';
				else
					rysuj = 'n';
			}
			else if(id % 10 == 5) {
				if(id < 10)
					rysuj = 'Q';
				else
					rysuj = 'q';
			}
			else if(id % 10 == 6) {
				if(id < 10)
					rysuj = 'K';
				else
					rysuj = 'k';
			}
			
			cout << " " << rysuj << " " << "|";
		}
		cout << "|";

		//Wypisywanie legendy
		switch(i) {
			case 0:
				cout << "	Oznaczenia:";
				break;
			case 1:
				cout << "	male litery - pionki czarne";
				break;
			case 2:
				cout << "	R - wieza";
				break;
			case 3:
				cout << "	N - skoczek";
				break;
			case 4:
				cout << "	K - krol";
				break;
		}

		cout << endl;
		if(i < 7) {
			cout << "-++";
			for(int k = 0; k < 8; k++) {
				cout << "---+";
			}
			cout << "+";
		}
		else {
			cout << "=++";
			for(int k = 0; k < 8; k++) {
				cout << "===+";
			}
			cout << "+";
		}
		
		switch(i) {
			case 0:
				cout << "	DUZE LITERY - pionki biale";
				break;
			case 1:
				cout << "	P - pion";
				break;
			case 2:
				cout << "	B - goniec";
				break;
			case 3:
				cout << "	Q - hetman";
				break;
		}
		cout << endl;
	}
	
	cout << " ||";
	for(int i = 0; i < 8; i++) {
		cout << " " << char('A' + i) << " " << "|";
	}
	cout << "|";
	cout << endl << endl;
}

//Zamiana 2 pierwszych znakow wczytanego od gracza stringa na indeksy rzedow i kolumn tablicy
//Jesli dowolny indeks wykracza poza tablice(indeks = <0, 7>) zwracamy wartosc false 
bool string_na_pole(int** plansza, string pole_str, int* pole)
{
	int kolumna = int(tolower(pole_str[0])) - (int)'a';
	int rzad = (int)'8' - (int)pole_str[1];

	if(rzad >= 0 && rzad <= 7)
		pole[0] = rzad;
	else
		return 0;
		
	if(kolumna >= 0 && kolumna <= 7)
		pole[1] = kolumna;
	else
		return 0;	
	
	return 1;
}

string pole_na_string(int** plansza, int* pole)
{
	string pole_str;
	pole_str[0] = char(pole[1] + 65);
	pole_str[1] = char(56 - pole[0]);

	return pole_str;
}

//Zwraca typ pionka (0 jesli pionek jest nieprawidlowy)
int sprawdz_pionek(int** plansza, int* pole, bool czyj_ruch) 
{
	int kolumna = pole[1];
	int rzad = pole[0];
	
	int pionek = plansza[rzad][kolumna];
	if(pionek == 0)
		return 0;
	else if(pionek > 0 && pionek < 10 && czyj_ruch == 0)
		return pionek % 10;
	else if(pionek > 10 && czyj_ruch == 1)
		return pionek % 10;
	
	return 0;
}

//Zwraca nazwe pionka na wybranym polu
string nazwa_pola(int** plansza, int* pole) {
	int pole_doc_typ = plansza[pole[0]][pole[1]];
	if(pole_doc_typ == 0)
		return "Puste pole";
	else if(pole_doc_typ % 10 == 1)
		return "Pion";
	else if(pole_doc_typ % 10 == 2)
		return "Wieza";
	else if(pole_doc_typ % 10 == 3)
		return "Goniec";
	else if(pole_doc_typ % 10 == 4)
		return "Skoczek";
	else if(pole_doc_typ % 10 == 5)
		return "Hetman";
	else if(pole_doc_typ % 10 == 6)
		return "Krol";
	
	return "Puste pole";
}

void wybierz_rozpoczecie(int** plansza, int* pole_pocz, int* pole_doc, bool czyj_ruch) 
{
	int losuj = rand() % 4;

	if(czyj_ruch == 0) {
		switch(losuj) {
			case 0:
				pole_pocz[0] = 6;
				pole_pocz[1] = 4;
				pole_doc[0] = 4;
				pole_doc[1] = 4;
				break;
			case 1:
				pole_pocz[0] = 6;
				pole_pocz[1] = 3;
				pole_doc[0] = 4;
				pole_doc[1] = 3;
				break;
			case 2:
				pole_pocz[0] = 7;
				pole_pocz[1] = 6;
				pole_doc[0] = 5;
				pole_doc[1] = 5;
				break;
			case 3:
				pole_pocz[0] = 7;
				pole_pocz[1] = 1;
				pole_doc[0] = 5;
				pole_doc[1] = 2;
				break;
		}
	}
	else {
		switch(losuj) {
			case 0:
				pole_pocz[0] = 1;
				pole_pocz[1] = 4;
				pole_doc[0] = 3;
				pole_doc[1] = 4;
				break;
			case 1:
				pole_pocz[0] = 1;
				pole_pocz[1] = 3;
				pole_doc[0] = 3;
				pole_doc[1] = 3;
				break;
			case 2:
				pole_pocz[0] = 0;
				pole_pocz[1] = 6;
				pole_doc[0] = 2;
				pole_doc[1] = 5;
				break;
			case 3:
				pole_pocz[0] = 0;
				pole_pocz[1] = 1;
				pole_doc[0] = 2;
				pole_doc[1] = 2;
				break;
		}
	}
	
}


//GPU tworzenie grafu
void licz_graf_GPU(int* plansza, vector<int> &punkty, int depth, int max_depth, bool czyj_ruch, bool czy_wezel_niepusty)
{
	if(depth == max_depth) {
		int punkty_wezel = 0;
		int* h_punkty = new int[64];
		bool czyj_ruch_nowy = 0;
		if(depth % 2 == 0)
			czyj_ruch_nowy = czyj_ruch;
		else
			czyj_ruch_nowy = !czyj_ruch;

		if(czy_wezel_niepusty == 1) {
			int* h_ruchy = new int[64];
			int* d_punkty;
        	int* d_plansza;

			cudaMalloc((void**)&d_plansza, 64 * sizeof(int));
			cudaMalloc((void**)&d_punkty, 64 * sizeof(int));

			cudaMemcpy(d_plansza, plansza, 64 * sizeof(int), cudaMemcpyHostToDevice);
			kernelEvalPoints<<<1,64>>>(d_plansza, d_punkty, czyj_ruch_nowy);
			cudaDeviceSynchronize();
			cudaMemcpy(h_punkty, d_punkty, sizeof(int), cudaMemcpyDeviceToHost);
			punkty_wezel = h_punkty[0];

			delete[] h_punkty;
			cudaFree(d_punkty);
			cudaFree(d_plansza);
		}
		else {
			if(max_depth % 2 == 1)
				punkty_wezel = MINIMAX_MINVAL;
			else
				punkty_wezel = MINIMAX_MAXVAL;
		}
		punkty.push_back(punkty_wezel);	
	}
	else if(czy_wezel_niepusty == 0) {
		for(int i = 0; i < 32; i++) {
			licz_graf_GPU(plansza, punkty, depth + 1, max_depth, !czyj_ruch, 0);
		}
	}
	else {
		//Znajdowanie pionkow
		int* h_ruchy = new int[64];
		int* d_ruchy;
        int* d_plansza;
		cudaMalloc((void**)&d_plansza, 64 * sizeof(int));
		cudaMalloc((void**)&d_ruchy, 64 * sizeof(int));

		cudaMemcpy(d_plansza, plansza, 64 * sizeof(int), cudaMemcpyHostToDevice);

		vector<int> pionki;
		kernelFindPawns<<<1,64>>>(d_plansza, d_ruchy, czyj_ruch);
		//Sortowanie
		thrust::device_ptr<int> thrust_tab(d_ruchy);
		thrust::sort(thrust_tab, thrust_tab + 64); 
		for(int i = 0; i < 16; i++) {
			if(thrust_tab[i] < 64) {
				pionki.push_back(thrust_tab[i]);
			}
			else
				break;
		}

		//Szukanie mozliwych ruchow
		vector<int> ruchy_start;
		vector<int> ruchy_doc;
		
		for(int i = 0; i < pionki.size(); i++) {
			if(ruchy_start.size() >= 32)
				break;

			kernelCheckAllMoves<<<1,64>>>(d_plansza, d_ruchy, pionki[i], czyj_ruch);
			cudaDeviceSynchronize();
			//Sortowanie
			thrust::device_ptr<int> thrust_tab(d_ruchy);
			thrust::sort(thrust_tab, thrust_tab + 64); 
			for(int j = 0; j < 64; j++) {
				if(thrust_tab[j] < 64) {
					ruchy_start.push_back(pionki[i]);
					ruchy_doc.push_back(thrust_tab[j]);
				}
				else
					break;
			}
		}
		delete[] h_ruchy;
		cudaFree(d_ruchy);
		cudaFree(d_plansza);
		
		//Rozszerzanie grafu dla wyliczonych ruchow
		for(int i = 0; i < 32; i++) {
			if(i >= ruchy_start.size())
				licz_graf_GPU(plansza, punkty, depth + 1, max_depth, !czyj_ruch, 0);
			else{
				int start_id = ruchy_start[i];
				int doc_id = ruchy_doc[i];
				int pole_start_typ = plansza[start_id];
				int pole_doc_typ = plansza[doc_id];

				plansza[doc_id] = pole_start_typ;
				plansza[start_id] = 0;
				//Wybor przez bota hetmana w razie promocji
				if(pole_start_typ == 0 && (doc_id/8 == 0 || doc_id/8 == 7)) {
					int pole_prom = 0;
					if(czyj_ruch == 0)
						pole_prom = 5;
					else
						pole_prom = 15;
					plansza[doc_id] = pole_prom;
				}
					
				licz_graf_GPU(plansza, punkty, depth + 1, max_depth, !czyj_ruch, 1);
				plansza[doc_id] = pole_doc_typ;
				plansza[start_id] = pole_start_typ;
			}
		}
	}
}

void znajdz_najlepszy_ruch(int* plansza, vector<int> punkty, int* max_ruch, bool czyj_ruch)
{
	//Znajdowanie pionkow
	int* h_ruchy = new int[64];
	int* d_ruchy;
    int* d_plansza;
	cudaMalloc((void**)&d_plansza, 64 * sizeof(int));
	cudaMalloc((void**)&d_ruchy, 64 * sizeof(int));

	cudaMemcpy(d_plansza, plansza, 64 * sizeof(int), cudaMemcpyHostToDevice);

	vector<int> pionki;
	kernelFindPawns<<<1,64>>>(d_plansza, d_ruchy, czyj_ruch);
	//Sortowanie
	thrust::device_ptr<int> thrust_tab(d_ruchy);
	thrust::sort(thrust_tab, thrust_tab + 64); 
	for(int i = 0; i < 16; i++) {
		if(thrust_tab[i] < 64) {
			pionki.push_back(thrust_tab[i]);
		}
	else
		break;
	}

	//Szukanie mozliwych ruchow
	vector<int> ruchy_start;
	vector<int> ruchy_doc;
		
	for(int i = 0; i < pionki.size(); i++) {
		if(ruchy_start.size() >= 32)
			break;

		kernelCheckAllMoves<<<1,64>>>(d_plansza, d_ruchy, pionki[i], czyj_ruch);
		cudaDeviceSynchronize();
		//Sortowanie
		thrust::device_ptr<int> thrust_tab(d_ruchy);
		thrust::sort(thrust_tab, thrust_tab + 64); 
		for(int j = 0; j < 64; j++) {
			if(thrust_tab[j] < 64) {
				ruchy_start.push_back(pionki[i]);
				ruchy_doc.push_back(thrust_tab[j]);
			}
			else
				break;
		}
	}
	delete[] h_ruchy;
	cudaFree(d_ruchy);
	cudaFree(d_plansza);

	int max_val = MINIMAX_MINVAL;
	int max_id_start = 0;
	int max_id_doc = 0;
	for(int i = 0; i < MAX_MOVES; i++) {
		if(punkty[i] >= max_val) {
			max_val = punkty[i];
			max_id_start = ruchy_start[i];
			max_id_doc = ruchy_doc[i];
		}
	}

	max_ruch[0] = max_id_start;
	max_ruch[1] = max_id_doc;
}



//--------------------------------------------Main------------------------------------------------------------------
int main()
{
	srand(time(NULL));
	//Tworzenie dwuwymiarowej tablicy 8x8 odpowiadajacej za figury znajdujace sie na polach
	int** plansza = new int* [8];
	int* gracze = new int[2];
	for (int i = 0; i < 8; i++) {
		plansza[i] = new int[8];
	}
	resetuj_plansze(plansza, gracze);
	bool czyj_ruch = 0;	//0 - biale, 1 - czarne
	int runda = 1;
	
	while(true) {
		system("clear");
		cout << "Wpisz 'r' w celu zresetowania gry" << endl << endl << endl;
		
		//Rysowanie planszy i calego "GUI"
		rysuj_plansze(plansza);
		if(czyj_ruch == 0)
			cout << "Obecny ruch - gracz BIALY" << endl << endl;
		else
			cout << "Obecny ruch - gracz CZARNY" << endl << endl;

		int* h_ruchy = new int[64];
		int* d_ruchy;
        int* d_plansza;

		cudaMalloc((void**)&d_plansza, 64 * sizeof(int));
		cudaMalloc((void**)&d_ruchy, 64 * sizeof(int));

		int* plansza_vec = new int[64];
		for(int i = 0; i < 64; i++) {
			plansza_vec[i] = plansza[i/8][i%8];
		}

		cudaMemcpy(d_plansza, plansza_vec, 64 * sizeof(int), cudaMemcpyHostToDevice);
		kernelEvalPoints<<<1, 64>>>(d_plansza, d_ruchy, czyj_ruch);
		cudaDeviceSynchronize();
		cudaMemcpy(h_ruchy, d_ruchy, sizeof(int), cudaMemcpyDeviceToHost);

		//cout << "Obecna plansza punkty: " << h_ruchy[0] << endl;

		int* pole_poczatkowe = new int[2];
		int* pole_docelowe = new int[2];
		int pionek;

		//Runda gracza(człowieka)
		if(gracze[czyj_ruch] == 0) {
			//Pobieramy od gracza pole piona, ktory chce ruszyc i sprawdzamy,
			//czy gracz posiada pion na takim polu
			if(czyj_ruch == 0 || czyj_ruch == 1) {
				string pole_str = "";
				cout << "Podaj pole pionka:" << endl;
				cin >> pole_str;
				pole_poczatkowe = new int[2];
				if(tolower(pole_str[0]) == 'r') {
					cout << "Gra zostanie zresetowana!" << endl;
					system("pause");
					resetuj_plansze(plansza, gracze);
					czyj_ruch = 0;
					runda = 1;
					continue;
				}
				if(string_na_pole(plansza, pole_str, pole_poczatkowe) == 0) {
					cout << "Nieprawidlowe pole!" << endl;
					system("pause");
					continue;
				}
				
				pionek = sprawdz_pionek(plansza, pole_poczatkowe, czyj_ruch);
				if(pionek == 0) {
					cout << "Nie masz pionka na tym polu!" << endl;
					system("pause");
					continue;
				}

				int start = pole_poczatkowe[0] * 8  + pole_poczatkowe[1];
				cudaMemcpy(d_plansza, plansza_vec, 64 * sizeof(int), cudaMemcpyHostToDevice);
				kernelCheckAllMoves<<<1, 64>>>(d_plansza, d_ruchy, start, czyj_ruch);
				cudaDeviceSynchronize();
				cudaMemcpy(h_ruchy, d_ruchy, 64 * sizeof(int), cudaMemcpyDeviceToHost);

				vector<int> pionek_ruchy;
				for(int i = 0; i < 64; i++) {
					if(h_ruchy[i] < 64)
						pionek_ruchy.push_back(h_ruchy[i]);
				}

				if(pionek_ruchy.empty()) {
					cout << "Pionek ten nie moze wykonac zadnego ruchu!" << endl;
					system("pause");
					continue;
				}
				else {
					cout << "Mozliwe ruchy:" << endl;
					for(int i = 0; i < pionek_ruchy.size(); i++) {
						int* pole_temp = new int[2];
						pole_temp[0] = pionek_ruchy[i] / 8;
						pole_temp[1] = pionek_ruchy[i] % 8;
						string temp_str = pole_na_string(plansza, pole_temp);
						cout << temp_str[0] << temp_str[1] << " ";
						if(i % 5 == 0 && i != 0)
							cout << endl;
						delete []pole_temp;
					}
					cout << endl;
				}
		
				
				//Pobieramy od gracza pole, na ktore chce ruszyc sie poprzednio wybranym pionem
				//i sprawdzamy, czy ruch jest prawidlowy
				cout << "Podaj ruch pionka:" << endl;
				cin >> pole_str;
				pole_docelowe = new int[2];
				if(tolower(pole_str[0]) == 'r') {
					cout << "Gra zostanie zresetowana!" << endl;
					system("pause");
					resetuj_plansze(plansza, gracze);
					czyj_ruch = 0;
					runda = 1;
					continue;
				}
				else if(string_na_pole(plansza, pole_str, pole_docelowe) == 0) {
					cout << "Nieprawidlowe pole!" << endl;
					system("pause");
					continue;
				}
				else if(pole_docelowe[0] == pole_poczatkowe[0] && pole_docelowe[1] == pole_poczatkowe[1]) {
					cout << "Nieprawidlowe pole!" << endl;
					system("pause");
					continue;
				}

				cudaMemcpy(d_plansza, plansza_vec, 64 * sizeof(int), cudaMemcpyHostToDevice);
				start = pole_poczatkowe[0] * 8  + pole_poczatkowe[1];
				int koniec = pole_docelowe[0] * 8  + pole_docelowe[1];
				kernelCheckMove<<<1, 1>>>(d_plansza, d_ruchy, start, koniec, czyj_ruch);
				cudaDeviceSynchronize();
				cudaMemcpy(h_ruchy, d_ruchy, sizeof(int), cudaMemcpyDeviceToHost);

				if(h_ruchy[0] > 64) {
					cout << "Nie mozesz wykonac takiego ruchu!" << endl;
					system("pause");
					continue;
				}
			}
			
			//Sprawdzanie promocji piona, wybor figury przez gracza jesli nastapila ona
			if(pionek == 1) {
				bool promocja = 0;
				if(czyj_ruch == 0 && pole_docelowe[0] == 0)
					promocja = 1;
				else if(czyj_ruch == 1 && pole_docelowe[0] == 7)
					promocja = 1;
				
				if(promocja == 1) {
					int figura_promocja = 0;
					cout << "Nastapila promocja! Wybierz figure, do ktorej ma awansowac pion:" << endl;
					cout << "2 - wieza" << endl;
					cout << "3 - goniec" << endl;
					cout << "4 - skoczek" << endl;
					cout << "5 - hetman" << endl;
					cin >> figura_promocja;
					if(figura_promocja >= 2 && figura_promocja <= 5) {
						pionek = figura_promocja;
					}	
					else {
						cout << "Nieprawidlowo wybrana figura!" << endl;
						system("pause");
						continue;
					}
				}
			}
		}
		//Pierwsza runda AI - wybor pierwszego ruchu
		else if(runda == 1 || runda == 2) {
			wybierz_rozpoczecie(plansza, pole_poczatkowe, pole_docelowe, czyj_ruch);
			pionek = plansza[pole_poczatkowe[0]][pole_poczatkowe[1]];

			string str_pocz = pole_na_string(plansza, pole_poczatkowe);
			string str_doc = pole_na_string(plansza, pole_docelowe);
			cout << "Wybrany ruch to " << str_pocz[0] << str_pocz[1] << "->" <<
				str_doc[0] << str_doc[1] << endl;
			
			string pole_str;
			cout << "Wpisz 'r' aby zrestartowac rozgrywke, lub cokolwiek innego, żeby AI wykonało ruch:" << endl;
			cin >> pole_str;
			if(tolower(pole_str[0]) == 'r') {
				cout << "Gra zostanie zresetowana!" << endl;
				system("pause");
				resetuj_plansze(plansza, gracze);
				czyj_ruch = 0;
				runda = 1;
				continue;
			}
		}
		//Runda AI
		else {
			vector<int> punkty_minimax;
			int max_glebokosc = 3;
			unsigned int rozmiar_minimax = pow(MAX_MOVES, max_glebokosc);
			licz_graf_GPU(plansza_vec, punkty_minimax, 0, max_glebokosc, czyj_ruch, 1);
			int* host_points = new int[punkty_minimax.size()];
			for(int i = 0; i < punkty_minimax.size(); i++) {
				host_points[i] = punkty_minimax[i];
			}

			int* dev_points;
			int* dev_max;
			int* dev_min;
			
			//Alokacja pamieci
			cudaMalloc((void**)&dev_points, rozmiar_minimax * sizeof(int));
			cudaMalloc((void**)&dev_max, rozmiar_minimax * sizeof(int));
			cudaMalloc((void**)&dev_min, rozmiar_minimax * sizeof(int));

			cudaMemcpy(dev_points, host_points, rozmiar_minimax * sizeof(int), cudaMemcpyHostToDevice);

			int watki = MAX_MOVES;
			unsigned int bloki = 1024;
			unsigned int max_blocks = pow(2, 31) - 1;

			for(int i = max_glebokosc; i > 1; i--) {
				unsigned int rozmiar_wynik = pow(MAX_MOVES, i-1);
				bloki = min(max_blocks, rozmiar_wynik);
				if(i % 2 == 0) {
					if(i == max_glebokosc) {
						kernelMin<<<bloki, watki>>>(dev_min, dev_points, rozmiar_wynik);
						cudaDeviceSynchronize();
					}	
					else {
						kernelMin<<<bloki, watki>>>(dev_min, dev_max, rozmiar_wynik);
						cudaDeviceSynchronize();
					}
				}
				else {
					if(i == max_glebokosc) {
						kernelMax<<<bloki, watki>>>(dev_max, dev_points, rozmiar_wynik);
						cudaDeviceSynchronize();
					}	
					else {
						kernelMax<<<bloki, watki>>>(dev_max, dev_min, rozmiar_wynik);
						cudaDeviceSynchronize();
					}	
				}
			}

			
			vector<int> punkty_ostatni_wezel;
			if(max_glebokosc != 1) {
				cudaMemcpy(host_points, dev_min, MAX_MOVES * sizeof(int), cudaMemcpyDeviceToHost);
				for(int i = 0; i < MAX_MOVES; i++) {
					punkty_ostatni_wezel.push_back(host_points[i]);
					//cout << punkty_ostatni_wezel[i] << " ";
				}
			}
			else {
				for(int i = 0; i < MAX_MOVES; i++) {
					punkty_ostatni_wezel.push_back(punkty_minimax[i]);
					//cout << punkty_ostatni_wezel[i] << " ";
				}
			}

			int* najlepszy_ruch = new int[2];
			znajdz_najlepszy_ruch(plansza_vec, punkty_ostatni_wezel, najlepszy_ruch, czyj_ruch);
			pole_poczatkowe[0] = najlepszy_ruch[0] / 8;
			pole_poczatkowe[1] = najlepszy_ruch[0] % 8;
			pole_docelowe[0] = najlepszy_ruch[1] / 8;
			pole_docelowe[1] = najlepszy_ruch[1] % 8;
			pionek = plansza[pole_poczatkowe[0]][pole_poczatkowe[1]] % 10;
			//Promocja
			if(pionek == 1 && (pole_docelowe[0] == 0 || pole_docelowe[0] == 7))	
				pionek = 5;	

			string str_pocz = pole_na_string(plansza, pole_poczatkowe);
			string str_doc = pole_na_string(plansza, pole_docelowe);
			cout << "Najlepszy wykryty ruch to " << str_pocz[0] << str_pocz[1] << "->" <<
				str_doc[0] << str_doc[1] << endl;
			
			string pole_str;
			cout << "Wpisz 'r' aby zrestartowac rozgrywke, lub cokolwiek innego, żeby AI wykonało ruch:" << endl;
			cin >> pole_str;
			if(tolower(pole_str[0]) == 'r') {
				cout << "Gra zostanie zresetowana!" << endl;
				system("pause");
				resetuj_plansze(plansza, gracze);
				czyj_ruch = 0;
				runda = 1;
				continue;
			}

			delete[] najlepszy_ruch;
			cudaFree(dev_points);
			cudaFree(dev_min);
			cudaFree(dev_max);
		}	


		//Sprawdzanie szach mat, ewentualne resetowanie gry
		if(plansza[pole_docelowe[0]][pole_docelowe[1]] % 10 == 6) {
			cout << "=================================" << endl;
			if(czyj_ruch == 0)
				cout << "Koniec gry! Wygrywa gracz bialy!" << endl;
			else
				cout << "Koniec gry! Wygrywa gracz Czarny!" << endl;
			cout << "=================================" << endl;
			string str;
			cin >> str;
			system("pause");
			resetuj_plansze(plansza, gracze);
			czyj_ruch = 0;
			runda = 1;
			continue;
		}

		//Przesuwanie pionka na nowe pole
		if(czyj_ruch == 1)
			pionek += 10;
		plansza[pole_poczatkowe[0]][pole_poczatkowe[1]] = 0;
		plansza[pole_docelowe[0]][pole_docelowe[1]] = pionek;
		
		//Zmiana grajacego gracza
		czyj_ruch = !czyj_ruch;
		runda++;

		delete []pole_poczatkowe;
		delete []pole_docelowe;
		cudaFree(d_plansza);
		cudaFree(d_ruchy);
	}

	delete []gracze;
	return 0;
}