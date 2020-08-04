/*
 * @Author: 	yshhuang@foxmail.com
 * @Date: 2020-06-09 17:17:11
 * @LastEditors: 	yshhuang@foxmail.com
 * @LastEditTime: 2020-06-10 18:39:26
 * @FilePath: /demo/src/main/java/com/yshhuang/BinarySearch.java
 */

import java.util.Arrays;

public class BinarySearch {
   private BinarySearch() {
   }

   public static int rank(int key, int[] a) {
      int lo = 0;
      int hi = a.length - 1;
      while (lo <= hi) {
         int mid = lo + (hi - lo) / 2;
         if (key < a[mid]) {
            hi = mid - 1;
         } else if (key > a[mid]) {
            lo = mid + 1;
         } else{
            return mid;
         }
      }
      return -1;
   }

   public static void main(String[] args) {
      int[] whitelist = In.readInts(args[0]);
   }
}
