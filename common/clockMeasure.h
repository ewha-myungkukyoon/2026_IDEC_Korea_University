/*
 * =====================================================================================
 *
 *       Filename:  clockMeasure.h
 *
 *    Description:  Simple wall-clock timer utility used across example
 *                  codes to measure and print elapsed CPU/GPU execution
 *                  time (supports pause/resume and accumulates across
 *                  multiple iterations).
 *
 *        Version:  1.0
 *        Created:  07/14/2026 01:10:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#pragma once
#include <stdio.h>
#include <time.h>
#include <string>

const int64_t nanoToSecTime=1000000000;

class clockMeasure {
 private:
  int64_t count;
  int64_t time;
  struct timespec begin, end;
  std::string name;

 public:
  clockMeasure(const std::string input) : count(0), time(0), name(input) {}
  ~clockMeasure() {}

  void clockReset() {
    time = 0;
    count = 0;
  }
  void clockResume() { clock_gettime(CLOCK_MONOTONIC, &begin); }
  void clockPause() {
    count++;
    clock_gettime(CLOCK_MONOTONIC, &end);
    time += (int64_t)(end.tv_sec - begin.tv_sec) * nanoToSecTime +
	    (int64_t)(end.tv_nsec - begin.tv_nsec);
  }
  //MK: Average time per resume/pause cycle in milliseconds, for callers that need the raw number (e.g. building a speedup table) instead of just the printed summary
  double getAvgTimeMs() {
    if (count == 0) return 0.0;
    return (double)time / (1000000.0 * count);
  }
  void clockPrint() {
    printf("[%s]", name.c_str());
    printf("clockMeasure[");
    if (count == 0) {
      printf("*None*]\n");
      return;
    }
    if (time < 1000) {
      printf("Total Time: \t%lf (ns), \tAvg Time: \t%lf (ns), \tCount: \t%ld",
	     (double)time, (double)time / count, count);
    } else if (time < 1000000) {
      printf("Total Time: \t%lf (us), \tAvg Time: \t%lf (us), \tCount: \t%ld",
	     (double)time / 1000, (double)time / (1000 * count), count);
    } else if (time < nanoToSecTime) {
      printf("Total Time: \t%lf (ms), \tAvg Time: \t%lf (ms), \tCount: \t%ld",
	     (double)time / 1000000, (double)time / (1000000 * count), count);
    } else {
      printf("Total Time: \t%lf (s), \tAvg Time: \t%lf (s), \tCount: \t%ld",
	     (double)time / nanoToSecTime, (double)time / (nanoToSecTime * count),
	     count);
    }
    printf("]\n");
  }
};
