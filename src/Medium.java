import com.sun.source.tree.Tree;

import java.util.*;

public class Medium {
    public class ListNode{
        int val;
        ListNode next;
        ListNode(){}
        ListNode(int val){
            this.val = val;
        }
        ListNode(int val, ListNode next){
            this.val = val;
            this.next = next;
        }
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2){

        if(l1 == null)
            return l2;
        if(l2 == null)
            return l1;

        int sum = 0;
        int carry = 0;
        ListNode result = new ListNode(0);
        ListNode curr = result;

        while(l1 != null || l2 != null){
            sum = (l1 != null ? l1.val : 0) + (l2 != null ? l2.val : 0) + carry;

            carry = sum/10;

            result.next = new ListNode(sum % 10);
            result = result.next;

            if(l1 != null)
                l1 = l1.next;
            if(l2 != null)
                l2 = l2.next;

        }

        if(carry > 0)
            result.next = new ListNode(carry);

        return curr.next;
    }

    public int lengthOfLongestSubstring(String s){
        // sliding window problem
//        int i = 0, j = 0;
//        int max = 0;
//        HashSet<Character> set = new HashSet<>();
//
//        while(j < s.length()){
//            if(!set.contains(s.charAt(j))){
//                set.add(s.charAt(j));
//                j++;
//                max = Math.max(max, set.size());
//            } else {
//                set.remove(s.charAt(i));
//                i++;
//            }
//        }
//
//        return max;

        //Map
        int ans = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        int i = 0;
        for(int j = 0; j < s.length(); j++){
            if(map.containsKey(s.charAt(j))){
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return ans;
    }

    public static int myAtoi(String s){
        if(s.isEmpty())
            return 0;
        int i = 0, sign = 1, base = 0;
        while(i < s.length() && s.charAt(i) == ' ')
            i++;

        if (i >= s.length()) return 0;
        if(s.charAt(i) == '-' || s.charAt(i) == '+')
            sign = s.charAt(i++) == '-' ? -1 : 1;

        while(i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9'){

            if(base > Integer.MAX_VALUE/10 || (base == Integer.MAX_VALUE/10 && s.charAt(i) - '0' > 7) )
                return (sign == 1) ? Integer.MAX_VALUE : Integer.MIN_VALUE;

            base = base * 10 + (s.charAt(i++) - '0');
        }

        return base * sign;

    }

    public int expandFromMiddle(String s, int left, int right){
        if(s == null || left > right)
            return 0;

        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
        }

        return right - left - 1;
    }
    public String longestPalindrome(String s){
        //O(n) time and O(1) space approach
//        if(s == null || s.length() < 1)
//            return "";
//        int start = 0;
//        int end = 0;
//
//        for(int i = 0; i < s.length(); i++){
//            int len1 = expandFromMiddle(s, i, i);
//            int len2 = expandFromMiddle(s, i, i+1);
//            int len = Math.max(len1, len2);
//            if(len > end - start){
//                start = i - ((len - 1)/2);
//                end = i + (len/2);
//            }
//        }
//        return s.substring(start, end + 1);

        //DP
        int n =s.length();
        String res = null;
        int palindromeStartsAt = 0, maxLen = 0;

        boolean[][] dp = new boolean[n][n];

        for(int i = n - 1; i >= 0; i-- ){
            for(int j = i ; j < i; j++){
                dp[i][j] = (s.charAt(i) == s.charAt(j)) && (j - i < 3 || dp[i+1][j-1]);

                if(dp[i][j] && (j - i + 1) > maxLen){
                    palindromeStartsAt = i;
                    maxLen = j - i  + 1;
                }
            }
        }

        return s.substring(palindromeStartsAt, palindromeStartsAt + maxLen);


    }

    public static String convert(String s, int numRows){
        char[] c = s.toCharArray();
        int len = c.length;
        StringBuffer[] sb = new StringBuffer[numRows];
        for(int i = 0; i < sb.length; i++){
            sb[i] = new StringBuffer();
        }
        int i = 0;
        while(i < len){
            for(int idx = 0; idx < numRows && i < len; idx++){
                sb[idx].append(c[i++]);
            }

            for(int idx = numRows-2; idx >= 1 && i < len; idx--){
                sb[idx].append(c[i++]);
            }
        }
        for(int idx = 1; idx < sb.length; idx++){
            sb[0].append(sb[idx]);
        }

        return sb[0].toString();
    }

    public static int maxArea(int[] height){
       int maxArea = 0;
       int left = 0, right = height.length - 1;
       while(left < right){
           maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * (right - left));
           if(height[left] < height[right])
               left++;
           else if(height[left] == height[right]){
               left++;
               right--;
           }
           else
               right--;
       }

       return maxArea;

    }

    public String intToRoman(int num){
        TreeMap<Integer, String> map = new TreeMap<>();
        map.put(1000, "M");
        map.put(900, "CM");
        map.put(500, "D");
        map.put(400, "CD");
        map.put(100, "C");
        map.put(90, "XC");
        map.put(50, "L");
        map.put(40, "XL");
        map.put(10, "X");
        map.put(9, "IX");
        map.put(5, "V");
        map.put(4, "IV");
        map.put(1, "I");

        StringBuilder sb = new StringBuilder();
        while(num != 0){
            int floor = map.floorKey(num);
            sb.append(map.get(floor));
            num -= floor;
        }
        return sb.toString();
    }

    public static void dfs(int s, String digits, StringBuilder sb, List<String> res, Map<Character, String> map){

        if(s == digits.length()){
            res.add(sb.toString());
            return;
        }

        String str = map.get(digits.charAt(s));
        for(int i = 0; i < str.length(); i++){
            sb.append(str.charAt(i));
            dfs(s + 1, digits, sb, res, map);
            sb.setLength(sb.length() - 1);
        }
    }
    public static List<String> letterCombinations(String digits){
        if(digits == null || digits.length() == 0)
            return new ArrayList<>();

        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");

        List<String> res = new ArrayList<>();
        dfs(0, digits, new StringBuilder(), res, map);

        return res;
    }

    public static List<List<Integer>> threeSum(int[] nums){
        List<List<Integer>> ans = new LinkedList<>();
        if(nums.length < 3)
            return ans;
        Arrays.sort(nums);

        for(int i = 0; i < nums.length - 2; i++){
//            if(nums[i] > 0)
//                break;
            if(i == 0 || (i > 0 && nums[i] != nums[i-1])){
                int low = i +1, high = nums.length - 1, sum = 0 - nums[i];
                while(low < high){
                    if(nums[low] + nums[high] == sum){
                        ans.add(Arrays.asList(nums[i], nums[low], nums[high]));
                        while(low < high && nums[low] == nums[low+1])
                            low++;
                        while(low < high && nums[high] == nums[high - 1])
                            high--;
                        low++;
                        high--;
                    }
                    else if(nums[low] + nums[high] < sum)
                        low++;
                    else
                        high--;
                }
            }

        }

        return ans;
    }

    public ListNode removeNthFromEnd(ListNode head, int n){
        ListNode temp = new ListNode(0);
        temp.next = head;
        ListNode a = temp, b = temp;
        while(n > 0){
            b = b.next;
            n--;
        }
        while(b.next != null){
            a = a.next;
            b = b.next;
        }
        a.next = a.next.next;
        return temp.next;
    }

    public void backtrack(List<String> ans, String str, int open, int close, int max){
        //base case
        if(str.length() == max * 2){
            ans.add(str);
            return;
        }
        if(open < max)
            backtrack(ans, str + "(", open+1, close, max);
        if(close < open)
            backtrack(ans, str + ")", open, close+1, max);

    }
    public List<String> generateParenthesis(int n){
        List<String> ans = new ArrayList<>();
        backtrack(ans, "", 0, 0, n);
        return ans;
    }

    public int search(int[] nums, int target){
        int n = nums.length;
        int low = 0, high = n - 1;
        while(low <= high){
            int mid = low + (high - low)/2;
            if(nums[mid] == target)
                return mid;
            if(nums[low] <= nums[mid]){
                if(nums[low] <= target && target < nums[mid])
                    high = mid - 1;
                else
                    low = mid + 1;
            } else{
                if(nums[mid] < target && target <= nums[high])
                    low = mid + 1;
                else
                    high = mid - 1;
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target){
        if(nums == null || nums.length == 0)
            return new int[]{-1,-1};
        int found = -1;
        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left)/2;
            if(nums[mid] == target){
                found = mid;
                break;
            }else if(nums[mid] > target)
                right = mid - 1;
            else
                left = mid + 1;
        }
        if(found == -1)
            return new int[]{-1,-1};
        int low = found, high = found;
        for(; low >= 0 && nums[low] == target; low--);
        for(; high < nums.length && nums[high] == target; high++);

        return new int[]{low + 1, high - 1};
    }

    public void islandsDFS(char[][] grid, int i, int j){

        if(i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0')
            return;

        grid[i][j] = '0';
        islandsDFS(grid, i - 1, j);
        islandsDFS(grid, i + 1, j);
        islandsDFS(grid, i, j - 1);
        islandsDFS(grid, i, j + 1);
    }
    public int numIslands(char[][] grid){
        int count = 0;

        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[i].length; j++){
                if(grid[i][j] == '1'){
                    count++;
                    islandsDFS(grid, i, j);
                }
            }
        }
        return count;
    }

    public ListNode oddEvenList(ListNode head){
        if(head == null || head.next == null)
            return head;

        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even;

        while(even != null && even.next != null){
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms){

        boolean[] visited = new boolean[rooms.size()];
        visited[0] = true;

        Stack<Integer> stack = new Stack<>();
        stack.push(0);

        while(!stack.isEmpty()){
            int temp = stack.pop();
            for(int room: rooms.get(temp)){
                if(!visited[room]) {
                    visited[room] = true;
                    stack.push(room);
                }
            }
        }

        for(boolean v: visited)
            if(!v)
                return false;

        return true;
    }
    public class TreeNode {
         int val;
         TreeNode left;
         TreeNode right;
         TreeNode() {}
         TreeNode(int val) { this.val = val; }
         TreeNode(int val, TreeNode left, TreeNode right) {
             this.val = val;
             this.left = left;
             this.right = right;
        }
    }

    public void func(TreeNode root, List<Integer> ans, int level){
        if(root == null)
            return;

        if(level == ans.size())
            ans.add(root.val);
        else
            ans.set(level, Math.max(ans.get(level), root.val));

        func(root.left, ans, level + 1);
        func(root.right, ans, level + 1);
    }
    public List<Integer> largestValue(TreeNode root){

        List<Integer> ans = new ArrayList<>();
        func(root, ans, 0);
        return ans;
    }

    public TreeNode pruneTree(TreeNode root){
        if(root == null)
            return null;
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);

        if(root.left == null && root.right == null && root.val == 0)
            return null;

        return root;
    }

    public static boolean isNStraightHand(int[] hand, int groupSize){

        TreeMap<Integer, Integer> counts = new TreeMap<>();
        for(int card: hand)
            counts.put(card, counts.getOrDefault(card, 0) + 1);
        while(counts.size() > 0){
            int first = counts.firstKey();
            for(int i = first; i < first + groupSize; i++){

                if(!counts.containsKey(i))
                    return false;
                int count = counts.get(i);
                if(count == 1)
                    counts.remove(i);
                else
                    counts.put(i, --count);
            }
        }
        return true;
    }
    public ListNode swapPairs(ListNode head){
        //recursive
//        if(head == null || head.next == null)
//            return head;
//
//        ListNode tmp1 = head;
//        ListNode tmp2 = head.next;
//
//        tmp1.next = tmp2.next;
//        tmp2.next = tmp1;
//
//        tmp1.next = swapPairs(tmp1.next);
//
//        return tmp2;

        //iterative
        ListNode temp = new ListNode(0);
        temp.next = head;
        ListNode curr = temp;

        while(curr != null && curr.next != null){
            ListNode first = curr.next;
            ListNode sec = curr.next.next;

            first.next = sec.next;
            curr.next = sec;
            sec.next = first;
            curr = curr.next.next;
        }

        return temp.next;
    }
    public int findBottomLeftValue(TreeNode root){

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        TreeNode temp = null;
        while(!q.isEmpty()){
            temp = q.poll();
            if(temp.right != null)
                q.offer(temp.right);
            if(temp.left != null)
                q.offer(temp.left);

        }
        return temp.val;
    }

    public ListNode partition(ListNode head, int x){
        ListNode before = new ListNode(0);
        ListNode after = new ListNode(0);
        ListNode curr1 = before, curr2 = after;

        while(head != null){
            if(head.val < x){
                curr1.next = head;
                curr1 = curr1.next;
            }else{
                curr2.next = head;
                curr2 = curr2.next;
            }
            head = head.next;
        }

        curr2.next = null;
        curr1.next = after.next;

        return before.next;
    }

    int res;
    public int distributeCoins(TreeNode root){
        res = 0;
        dfs(root);
        return res;
    }

    public int dfs(TreeNode root){

        if(root == null)
            return 0;
        int left = dfs(root.left);
        int right = dfs(root.right);

        res += Math.abs(left) + Math.abs(right);
        return root.val + left + right - 1;
    }

    public boolean isCompleteTree(TreeNode root){

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while(q.peek() != null){
            TreeNode temp = q.poll();
            q.offer(temp.left);
            q.offer(temp.right);
        }
        while(!q.isEmpty() && q.peek() == null)
            q.poll();

            return q.isEmpty();
    }
    public void flatten(TreeNode root){
        //iterative way
       if(root == null)
           return;
       Stack<TreeNode> st = new Stack<>();
       st.push(root);

       while(!st.isEmpty()){
           TreeNode temp = st.pop();
           if(temp.right != null)
               st.push(temp.right);
           if(temp.left != null)
               st.push(temp.left);
           if(!st.isEmpty())
               temp.right = st.peek();

           temp.left = null;
       }

//        if(root == null)
//            return;
//
//        TreeNode left = root.left;
//        TreeNode right = root.right;
//
//        root.left = null;
//
//        flatten(left);
//        flatten(right);
//
//        root.right = left;

    }

    public void rightView(TreeNode root, List<Integer> ans, int depth){
        if(root == null)
            return;

        if(ans.size() == depth)
            ans.add(root.val);

        rightView(root.right, ans, depth + 1);
        rightView(root.left, ans, depth + 1);
    }
    public List<Integer> rightSideView(TreeNode root){
        List<Integer> ans = new ArrayList<>();
        rightView(root, ans, 0);
        return ans;
    }

    public ListNode merge(ListNode left, ListNode right){
        ListNode l = new ListNode(0), curr = l;

        while(left != null && right != null){
            if(left.val < right.val){
                curr.next = left;
                left = left.next;
            }
            else{
                curr.next = right;
                right = right.next;
            }

            curr = curr.next;
        }
        if(left != null)
            curr.next = left;
        if(right != null)
            curr.next = right;

        return l.next;

    }
    public ListNode sortList(ListNode head){
        if(head == null || head.next == null)
            return head;

        ListNode prev = null;
        ListNode slow = head;
        ListNode fast = head;

        while(fast != null && fast.next != null){
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        prev.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(slow);

        return merge(left, right);
    }

    public List<Integer> spiralOrder(int[][] matrix){

        List<Integer> ans = new ArrayList<>();
        int top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;
        while(top <= bottom && left <= right){

            for(int i = left; i <= right; i++)
                ans.add(matrix[top][i]);
            top++;

            for(int i = top; i <= bottom; i++)
                ans.add(matrix[i][right]);
            right--;
            if(top <= bottom){
                for(int i = right; i >= left; i--)
                    ans.add(matrix[bottom][i]);
                bottom--;
            }
            if(left <= right){
                for(int i = bottom; i >= top; i--)
                    ans.add(matrix[i][left]);
                left++;
            }
        }
        return ans;
    }

    public int[][] generateMatrix(int n){
        int[][] ans = new int[n][n];

        int rowBegin = 0, rowEnd = n-1, colBegin = 0, colEnd = n-1;
        int counter = 1;

        while(rowBegin <= rowEnd && colBegin <= colEnd){

            for(int i = colBegin; i <= colEnd; i++)
                ans[rowBegin][i] = counter++;

            rowBegin++;

            for(int i = rowBegin; i <= rowEnd; i++)
                ans[i][colEnd] = counter++;

            colEnd--;

            if(rowBegin <= rowEnd){
                for(int i = colEnd; i >= colBegin; i--)
                    ans[rowEnd][i] = counter++;

            }
            rowEnd--;
            if(colBegin <= colEnd){
                for(int i = rowEnd ;i >= rowBegin; i--)
                    ans[i][colBegin] = counter++;
            }

            colBegin++;
        }

        return ans;
    }

    public int minSubArrayLen(int target, int[] nums){

        int sum = 0, min = Integer.MAX_VALUE;
        int j = 0;

        for(int i = 0; i < nums.length; i++){
            sum += nums[i];
            while(sum >= target){
                min = Math.min(min, i - j + 1);
                sum -= nums[j++];
            }
        }

        return min == Integer.MAX_VALUE ? 0 : min;
    }

    public boolean searchMatrix(int[][] matrix, int target){
        if(matrix.length == 0)
            return false;

        int row = matrix.length;
        int col = matrix[0].length;

        int left = 0, right = row * col - 1;

        while(left <= right){
            int mid = left + (right - left)/2;
            int mid_ele = matrix[mid / col][mid % col];
            if(mid_ele == target)
                return true;
            else if(mid_ele > target)
                right = mid - 1;
            else
                left = mid + 1;
        }

        return false;
    }

    public void permuteBackTrack(List<List<Integer>> result, int[] nums, List<Integer> currList, int index){
        if(currList.size() == nums.length){
            result.add(currList);
            return;
        }

        int n = nums[index];
        for(int i = 0; i <= currList.size(); i++){
            List<Integer> temp = new ArrayList<>(currList);
            temp.add(i, n);
            permuteBackTrack(result, nums, temp, index+1);
        }
    }
    public List<List<Integer>> permute(int[] nums){
        List<List<Integer>> result = new ArrayList<>();

        permuteBackTrack(result, nums, new ArrayList<Integer>(), 0);

        return result;
    }


        public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>());

        for(int n: nums){
            int size = result.size();
            for(int i = 0; i < size; i++){
                List<Integer> subset = new ArrayList<>(result.get(i));
                subset.add(n);
                result.add(subset);
            }
        }

        return result;

    }

    public static int count = 0;
    public static int number = 0;
    public void kthSmallestfunc(TreeNode node){
        if(node.left != null)
            kthSmallestfunc(node.left);

        count--;

        if(count == 0){
            number = node.val;
            return;
        }

        if(node.right != null)
            kthSmallestfunc(node.right);
    }
    public int kthSmallest(TreeNode root, int k){

        count = k;
        kthSmallestfunc(root);

        return number;
    }

    public int areaOfIsland(int[][] grid, int i, int j){

        if(i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1){
            grid[i][j] = 0;
            return 1 + areaOfIsland(grid, i + 1, j) + areaOfIsland(grid,i - 1, j) + areaOfIsland(grid, i, j - 1) + areaOfIsland(grid, i, j + 1);
        }
        return 0;
    }
    public int maxAreaOfIsland(int[][] grid){
        int max_area = 0;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    max_area = Math.max(max_area, areaOfIsland(grid, i, j));
                }
            }
        }

        return max_area;
    }

    public int totalFruit(int[] tree) {
        int last = -1, second_last = -1, last_count = 0;
        int curr_max = 0, max = 0;

        for(int fruit: tree){

            if(fruit == last || fruit == second_last)
                curr_max++;
            else
                curr_max = last_count + 1;

            if(fruit == last)
                last_count++;
            else{
                last_count = 1;
            }

            if(fruit != last){
                second_last = last;
                last = fruit;
            }

            max = Math.max(max, curr_max);
        }
        return max;
    }

    public static void main(String[] args) {

        //System.out.println(convert("PAYPALISHIRING", 3));
        //System.out.println(myAtoi(" "));
        System.out.println(isNStraightHand(new int[]{1,2,3,6,2,3,4,7,8}, 3));



    }
}
