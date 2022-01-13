import java.util.*;

public class Hard {
    public int countVowelPermutation(int n){

        if(n == 1)
            return 5;
        if(n == 2)
            return 10;
        int mod = (int)1e9 + 7;

        long[][] dp = new long[n + 1][5];

        for(int i = 0; i < 5; i++)
            dp[1][i] = 1;

        for(int i = 1; i < n; i++){

            //a
            dp[i + 1][0] = (dp[i][1] + dp[i][2] + dp[i][4]) % mod;

            //e
            dp[i + 1][1] = (dp[i][0] + dp[i][2]) % mod;

            //i
            dp[i + 1][2] = (dp[i][1] + dp[i][3]) % mod;

            //o
            dp[i + 1][3] = (dp[i][2]) % mod;

            //u
            dp[i + 1][4] = (dp[i][2] + dp[i][3]) % mod;
        }
        long ans = 0;
        for(int i = 0; i < 5; i++){
            ans = (ans + dp[n][i]) % mod;
        }

        return (int)ans;

    }

    public static int maxSatisfaction(int[] satisfaction){

        Arrays.sort(satisfaction);
        int curr = 0, ans = 0, tempSum = 0;

        for(int i = satisfaction.length - 1; i >= 0; i--){
            tempSum += satisfaction[i];
            curr += tempSum;

            ans = Math.max(ans, curr);
        }

        return ans;
    }



    public int minFallingPathSumII(int[][] arr) {

        if(arr == null)
            return 0;
        int m = arr.length, n = arr[0].length, min = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int[][] dp = new int[m][n];

        for(int i = 0; i < n; i++){
            dp[0][i] = arr[0][i];
            if(dp[0][i] <= min){
                min2 = min;
                min = dp[0][i];
            } else
                min2 = Math.min(dp[0][i], min2);
        }

        for(int i = 1; i < m; i++){
            int currMin = Integer.MAX_VALUE, currMin2 = Integer.MAX_VALUE;

            for(int j = 0; j < n; j++){

                dp[i][j] = (dp[i - 1][j] == min) ? arr[i][j] + min2: min + arr[i][j];

                if(dp[i][j] <= currMin){
                    currMin2 = currMin;
                    currMin = dp[i][j];
                } else
                    currMin2 = Math.min(currMin2, dp[i][j]);
            }

            min = currMin;
            min2 = currMin2;
        }
        return min;
    }

    public int trap(int[] height) {

        int n = height.length;
        if( n <= 2)
            return 0;
        int[] leftMax = new int[n];
        int[] rightMax = new int[n];

        leftMax[0] = height[0];
        rightMax[n-1] = height[n-1];

        for(int i = 1, j = n-2; i < n; i++, j--){
            leftMax[i] = Math.max(leftMax[i-1], height[i]);
            rightMax[j] = Math.max(rightMax[j + 1], height[j]);
        }

        int totalWater = 0;
        for(int k = 1; k < n - 1; k++){
            int water = Math.min(leftMax[k -1], rightMax[k + 1] ) - height[k];
            totalWater += (water > 0) ? water : 0;
        }

        return totalWater;
    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {

        int n = startTime.length;
        int[][] jobs = new int[n][3];

        for(int i = 0; i < n; i++){
            jobs[i] = new int[]{ startTime[i], endTime[i], profit[i]};
        }

        Arrays.sort(jobs, (a, b) -> a[1] - b[1]);

        TreeMap<Integer, Integer> dp = new TreeMap<>();
        dp.put(0, 0);

        for(int[] job: jobs){
            int cur = dp.floorEntry(job[0]).getValue() + job[2];
            if(cur > dp.lastEntry().getValue()){
                dp.put(job[1], cur);
            }
        }

        return dp.lastEntry().getValue();

//        int[] dp = new int[startTime.length];
//        dp[0] = jobs[0][2];
//
//        for(int i = 1; i < profit.length; i++){
//            dp[i] = dp[i-1];
//            for(int j = i-1; j >= 0; j--){
//
//                if(jobs[j][1] <= jobs[i][0]){
//                    dp[i] = Math.max(dp[i], dp[j] + jobs[i][2]);
//                    break;
//                }
//                dp[i] = Math.max(dp[i], jobs[i][2]);
//            }
//        }
//        return dp[profit.length - 1];
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {

        HashSet<String> set = new HashSet<>(wordList);
        if(!set.contains(endWord)){
            return 0;
        }
        Queue<String> q = new LinkedList<>();
        q.offer(beginWord);
        int level = 1;

        while(!q.isEmpty()){

            int size = q.size();
            for(int i = 0; i < size; i++){
                String str = q.poll();
                char[] words = str.toCharArray();

                for(int j = 0; j < words.length; j++){

                    char og_ch = words[j];

                    for(char c = 'a'; c <= 'z'; c++){
                        if(words[j] == c)
                            continue;
                        words[j] = c;
                        String next = new String(words);
                        if(next.equals(endWord)){
                            return level + 1;
                        }
                        if(set.contains(next)){
                            q.offer(next);
                            set.remove(next);
                        }
                    }
                    words[j] = og_ch;

                }
            }

            level++;
        }

        return 0;

    }


    public static void main(String[] args) {

//        int[] satisfaction = {4,3,2};
//        System.out.println(maxSatisfaction(satisfaction));

        System.out.println(new Hard().jobScheduling(new int[]{1,2,3,3}, new int[]{3,4,5,6}, new int[]{50, 10,40,70}));
    }
}
