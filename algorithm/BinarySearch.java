/*
 * @Author: 	yshhuang@foxmail.com
 * @Date: 2020-06-09 17:17:11
 * @LastEditors: 	yshhuang@foxmail.com
 * @LastEditTime: 2020-06-09 18:43:12
 * @FilePath: /algorithm/BinarySearch.java
 */


import java.util.Arrays;

public class BinarySearch {
   private BinarySearch() {
   }

   public static int rank(final int key, final int[] a) {
      int lo = 0;
      int hi = a.length - 1;
      while (lo <= hi) {
         final int mid = lo + (hi - lo) / 2;
         if (key < a[mid]) {
            hi = mid - 1;
         } else if (key > a[mid]) {
            lo = mid + 1;
         } else {
            return mid;
         }
      }
      return -1;
   }

   public static void main(final String[] args) {
      final int[] whitelist = In.readInts(args[0]);
   }
}
