/* testblas.c - test program for the cblas_dgemv function
 *
 * Copyright (C) 2004  Jochen Voss.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * $Id: testblas.c 5698 2004-05-12 18:17:53Z voss $
 */

#include <stdio.h>
#include <cblas.h>

double m[] = {
  3, 1, 3,
  1, 5, 9,
  2, 6, 5
};

double x[] = {
  -1, -1, 1
};

double y[] = {
  0, 0, 0
};

int
main()
{
  int i, j;

  for (i=0; i<3; ++i) {
    for (j=0; j<3; ++j) printf("%5.1f", m[i*3+j]);
    putchar('\n');
  }

  cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, m, 3,
	      x, 1, 0.0, y, 1);

  for (i=0; i<3; ++i)  printf("%5.1f\n", y[i]);

  return 0;
}
