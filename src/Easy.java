import java.util.*;

public class Easy {
    public static int[] twoSum(int[] nums, int target){
        //brute force approach
//        int ans[] = new int[2];
//        for(int i = 0; i < nums.length; i++){
//            for(int j = i + 1; j < nums.length; j++){
//                if(nums[i] + nums[j] == target){
//                    ans[0] = i;
//                    ans[1] = j;
//                }
//
//            }
//        }
//        return ans;

        //Efficient approach
        int[] ans = new int[2];
        Map<Integer, Integer> num_map = new HashMap<>();
        for(int i= 0; i < nums.length; i++){
            int ele = target - nums[i];
            if(num_map.containsKey(ele)){
                ans[0] = num_map.get(ele);
                ans[1] = i;
                break;
            }

            num_map.put(nums[i], i);
        }
        return ans;
    }

    public static int reverse(int x) {
        //1st approach
        boolean negative = false;

        if(x < 0){
            negative = true;
            x *= -1;
        }

        int reversedInteger = 0;

        while(x != 0){

            int rem = x % 10;
            x = x/10;


            if(reversedInteger > Integer.MAX_VALUE/10 || reversedInteger == Integer.MAX_VALUE/10 && rem  > 7)
                return 0;
            //don't really need as we convert negative integer to a positive one
            if(reversedInteger < Integer.MIN_VALUE/10 || reversedInteger == Integer.MIN_VALUE/10 && rem < -8 )
                return 0;

            reversedInteger = (reversedInteger * 10) + rem;

        }
        return negative ? (-1 * reversedInteger) : reversedInteger;
    }

    public boolean isPalindrome(int x) {
        if(x == 0)
            return true;

        if(x < 0 || x % 10 == 0)
            return false;

        int rev = 0;
        while(x > rev){
            int rem = x % 10;
            x = x/10;

            rev = (rev * 10) + rem;

        }
        if(x == rev || x == rev/10)
            return true;
        else
            return false;

    }
    public int romanToInt(String s) {

        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        int result = 0;

        for(int i = 0; i < s.length(); i++){

            if(i > 0 && map.get(s.charAt(i)) > map.get(s.charAt(i-1))){
                result += map.get(s.charAt(i)) - 2 * map.get(s.charAt(i-1));
            } else {
                result += map.get(s.charAt(i));
            }
        }
        return result;

    }
    public static String longestCommonPrefix(String[] strs) {

        //Leet code official solution

        if(strs.length == 0)
            return "";

        String prefix = strs[0];

        for(int i = 1; i < strs.length; i++){
            while(strs[i].indexOf(prefix) != 0){
               // System.out.println(strs[i].indexOf(prefix));
                prefix = prefix.substring(0, prefix.length()- 1);
            }
        }
        return prefix;


        //another solution
        /*
        String prefix = "";
        if(strs == null || strs.length == 0)
            return prefix;

        int index = 0;
        for(char c: strs[0].toCharArray()){
            for(int i = 1; i < strs.length; i++){
                if(index >= strs[i].length() || c != strs[i].charAt(index))
                    return prefix;
            }
            prefix += c;
            index++;
        }
        return prefix;
        */
    }
    public static boolean isValid(String s) {
        //brackets have to be even
        if(s.length() % 2 != 0)
            return false;

        Stack<Character> stack = new Stack<>();
        for(char c: s.toCharArray()){

            if(c == '(' || c == '{' || c == '[')
                stack.push(c);
            else if(c == ')' && !stack.isEmpty() && stack.peek() == '(')
                stack.pop();
            else if(c == '}' && !stack.isEmpty() && stack.peek() == '{')
                stack.pop();
            else if(c == ']' && !stack.isEmpty() && stack.peek() == '[')
                stack.pop();
            else
                return false;

        }

        return stack.isEmpty() ? true : false;

    }
    public class ListNode {
     int val;
     ListNode next;
     ListNode() {}
     ListNode(int val) { this.val = val; }
     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 }
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        //original approach
        /*
         if(l1 == null)
             return l2;
         if(l2 == null)
             return l1;
         if(l1.val < l2.val){
             l1.next = mergeTwoLists(l1.next, l2);
             return l1;
         } else {
             l2.next = mergeTwoLists(l1, l2.next);
             return l2;
         }
         */
         //new approach
        ListNode temp = new ListNode(0);
        ListNode curr_node = temp;
        while(l1 != null && l2 != null){
            if(l1.val < l2.val){
                curr_node.next = l1;
                l1 = l1.next;
            }
            else {
                curr_node.next = l2;
                l2 = l2.next;
            }
            curr_node = curr_node.next;
        }
        if(l1 != null){
            curr_node.next = l1;
            l1 = l1.next;
        }
        if(l2 != null){
            curr_node.next = l2;
            l2 = l2.next;
        }
        return temp.next;

    }
    public int removeDuplicates(int[] nums) {


        int i = 0, j = 1;
        int n = nums.length;
        for(j = 1; j < n; j++){
            if(nums[j] != nums[i]){
                i++;
                nums[i] = nums[j];

            }
        }
        return i+1;


    }
    public int removeElement(int[] nums, int val) {

        int i = 0;
        for(int j = 0; j < nums.length; j++){
            if(nums[j] != val){
                nums[i] = nums[j];
                i++;
            }
        }
        return i;
    }
    public int strStr(String haystack, String needle){
        //1st approach
//        int hLen = haystack.length();
//        int nLen = needle.length();
//
//        for(int i = 0; i <= hLen - nLen; i++){
//            int j = 0;
//            for(; j < nLen; j++){
//                if(needle.charAt(j) !=  haystack.charAt(i+j))
//                    break;
//            }
//            if(j == nLen)
//                return i;
//        }
//        return -1;

        //100% approach
        return haystack.indexOf(needle);
    }
    public int searchInsert(int[] nums, int target) {

//        // basic approach : O(n) time complexity
//         int n = nums.length;

//         for(int i = 0; i < n; i++){
//             if(nums[i] == target)
//                 return i;
//             if(nums[i] > target)
//                 return i;
//         }
//         return -1;

        // O(log n) approach
        int left = 0;
        int right = nums.length - 1;

        while(left <= right){
            int mid = left + (right - left)/2;

            if(nums[mid] == target)
                return mid;
            if(target > nums[mid])
                left = mid + 1;
            else
                right = mid - 1;


        }
        return left;
    }

    public int maxSubArray(int[] nums) {
        //O(n) approach
//         int n = nums.length;
//         int maxSum = Integer.MIN_VALUE, localSum = 0;

//         for(int i = 0; i < n; i++){

//             localSum += nums[i];
//             if(localSum > maxSum)
//                 maxSum = localSum;

//             if(localSum < 0)
//                 localSum = 0;
//         }
//         return maxSum;

        //O(log n) divide and conquer approach
        //1st approach
        // return func(nums, 0, nums.length - 1);
        //2nd approach
        return maxSubArr(nums, 0, nums.length - 1);
    }

    public int maxSubArr(int[] a, int left, int right){
        if(left ==  right)
            return a[left];
        int mid = left + (right - left)/2;

        int maxArrLeft = maxSubArr(a, left, mid);
        int maxArrRight = maxSubArr(a, mid + 1, right);
        int maxSingle = Math.max(maxArrLeft, maxArrRight);

        int sum = 0;
        int maxRight = Integer.MIN_VALUE;
        for(int i = mid + 1; i <= right; i++){
            sum += a[i];
            maxRight = Math.max(maxRight, sum);
        }

        sum = 0;
        int maxLeft = Integer.MIN_VALUE;
        for(int i = mid; i >= left; i--){
            sum += a[i];
            maxLeft = Math.max(maxLeft, sum);
        }
        return Math.max(maxSingle, maxLeft + maxRight);
    }

//     public int func(int[] nums, int left, int right){
//         if(left > right)
//             return Integer.MIN_VALUE;
//         if(left == right)
//             return nums[left];
//         int mid = left + (right - left)/2;
//         return Math.max(Math.max(func(nums, left, mid - 1), func(nums, mid + 1, right)), crossMid(nums, left, right));
//     }

//     public int crossMid(int[] nums, int left, int right){
//         int mid = left + (right - left)/2;

//         int lLeft = nums[mid], maxLeft = nums[mid];
//         for(int i = mid - 1; i >= left; i--){
//             lLeft += nums[i];
//             maxLeft = Math.max(maxLeft, lLeft);
//         }

//         int lRight = nums[mid], maxRight = nums[mid];
//         for(int i = mid + 1; i <= right; i++){
//             lRight += nums[i];
//             maxRight = Math.max(maxRight, lRight);
//         }

    //         return maxLeft + maxRight - nums[mid];
//     }

    public int lengthOfLastWord(String s) {
        char[] chars = s.toCharArray();
        int ans = 0;
        boolean word = false;;

        for(int i = chars.length - 1; i >= 0; i--){
            if(chars[i] == ' ' && word)
                break;
            else if(chars[i] == ' '){
                continue;
            }
            else{
                ans++;
                word = true;
            }
        }
        return ans;
    }

    public int[] plusOne(int[] digits){
        int idx = digits.length - 1;
        boolean carry = true;

        while(carry && idx >= 0){
            if(digits[idx] == 9){
                digits[idx] = 0;
                carry = true;
            } else{
                digits[idx] += 1;
                carry = false;
            }
            idx--;
        }

        if(digits[0] == 0){
            int[] results = new int[digits.length + 1];
            results[0] = 1;
            for(int i = 1; i <= digits.length; i++){
                results[i] = digits[i-1];
            }
            return results;
        }
        return digits;
    }

    public static String addBinary(String a, String b){
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;
        while(i >= 0 || j >= 0){
            int sum = carry;
            if(i >= 0)
                sum += a.charAt(i--) - '0';
            if(j >= 0)
                sum += b.charAt(j--) - '0';
            sb.insert(0, sum % 2);
            carry = sum / 2;
        }
        if(carry != 0)
            sb.insert(0, carry);

        return sb.toString();
    }

    public int mySqrt(int x){
//        if(x == 0)
//            return 0;
//        int left = 1, right = x, target = x;
//        while(left <= right){
//            int mid = left + (right - left)/2;
//            if(mid > 46340)
//                right = mid - 1;
//            else if(target == mid * mid)
//                return mid;
//            else if(target < mid * mid)
//                right = mid - 1;
//            else
//                left = mid + 1;
//        }
//        return right;

        return (int) Math.sqrt(x);
    }

    public int climbStairs(int n){
        int[] dp = new int[n+1];

        if(n == 0 || n == 1)
            return 1;

        dp[0] = 1;
        dp[1] = 1;


        for(int i = 2; i <= n; i++)
            dp[i] = dp[i-1] + dp[i-2];

        return dp[n];
    }

    public ListNode deleteDuplicates(ListNode head){
        if(head == null || head.next == null)
            return head;
        ListNode curr = head;
        while(curr.next != null){
            if(curr.val == curr.next.val)
                curr.next = curr.next.next;
            else
                curr = curr.next;
        }
        return head;
    }
    public void merge(int[] nums1, int m, int[] nums2, int n){
        int[] temp = new int[m];
        for(int i = 0; i < m; i++)
            temp[i] = nums1[i];

        int i = 0, j = 0, k = 0;
        while(i < m && j < n){
            if(temp[i] < nums2[j]){
                nums1[k] = temp[i];
                i++;
                k++;
            } else {
                nums1[k] = nums2[j];
                j++;
                k++;
            }
        }
        if(i == m){
            for(int p = j; p < n; p++) {
                nums1[k] = nums2[p];
                k++;
            }
        } else{
            for(int p = i; p < m; p++) {
                nums1[k] = temp[p];
                k++;
            }
        }
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
    List<Integer> list = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        if(root == null)
            return list;
        inorderTraversal(root.left);
        list.add(root.val);
        inorderTraversal(root.right);

        return list;
    }
    public boolean isSameTree(TreeNode p, TreeNode q) {

        if(p == null && q == null)
            return true;
        if(p == null || q == null)
            return false;
        if(p.val != q.val)
            return false;

        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public boolean isSymmetric(TreeNode root){
        if(root == null)
            return true;
        return isMirror(root.left, root.right);
    }
    public boolean isMirror(TreeNode p, TreeNode q){
        if(p == null && q == null)
            return true;
        if(p != null && q != null && p.val == q.val)
            return isMirror(p.left, q.right) && isMirror(p.right, q.left);

        return false;
    }

    public int maxDepth(TreeNode root){
        if(root == null)
            return 0;
        int lDepth = maxDepth(root.left);
        int rDepth = maxDepth(root.right);

        return 1 + Math.max(lDepth, rDepth);
    }
    public TreeNode sortedArrayToBST(int[] nums){
        return sortArrBSTfunc(nums, 0, nums.length - 1);
    }

    public TreeNode sortArrBSTfunc(int[] nums, int left, int right){
        if(left >  right)
            return null;
        int mid = left + (right - left)/2;
        TreeNode new_node = new TreeNode(nums[mid]);

        new_node.left = sortArrBSTfunc(nums, left, mid - 1);
        new_node.right = sortArrBSTfunc(nums, mid + 1, right);

        return new_node;
    }

    public int height(TreeNode root){
        if(root == null)
            return 0;
        return 1 + Math.max(height(root.left), height(root.right));
    }
    public boolean isBalanced(TreeNode root){
        if(root == null)
            return true;
        int lHeight = height(root.left);
        int rHeight = height(root.right);
        if(Math.abs(lHeight - rHeight) <= 1 && isBalanced(root.left) && isBalanced(root.right))
            return true;
        else
            return false;
    }

    public int minDepth(TreeNode root){
        if(root == null)
            return 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int level = 1;
        while(!q.isEmpty()){
            int size = q.size();
            for(int i = 0; i < size; i++){
                TreeNode curr = new TreeNode();
                if(curr.left == null && curr.right == null)
                    return level;
                if(curr.left != null)
                    q.offer(curr.left);
                if(curr.right != null)
                    q.offer(curr.right);
            }
            level++;
        }
        return level;
    }

    public boolean hasPathSum(TreeNode root, int targetSum){
        if(root == null)
            return false;
        if(root.left == null && root.right == null && targetSum - root.val == 0)
            return true;

        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }

    public List<List<Integer>> generate(int numRows) {
        //Pascal's Triangle
//        List<List<Integer>> ans = new ArrayList<>();
//        List<Integer> row = new ArrayList<>();
//        for(int i = 0; i < numRows; i++){
//            row.add(0, 1);
//            for(int j = 1; j < row.size() - 1; j++)
//                row.set(j, row.get(j) + row.get(j+1));
//            ans.add(new ArrayList<>(row));
//        }
//        return ans;

        List<List<Integer>> triangle = new ArrayList<>();

        List<Integer> first_row = new ArrayList<>();
        first_row.add(1);
        triangle.add(first_row);

        for(int i = 1; i < numRows; i++){
            List<Integer> prev_row = triangle.get(i-1);
            List<Integer> row = new ArrayList<>();

            row.add(1);
            for(int j = 1; j < i; j++){
                row.add(prev_row.get(j-1) + prev_row.get(j));
            }
            row.add(1);
            triangle.add(row);
        }
        return triangle;
    }
    public List<Integer> getRow(int rowIndex){
        List<Integer> ans = new ArrayList<>();
        for(int i = 0; i <= rowIndex; i++){
            ans.add(1);
            for(int j = i - 1; j > 0; j--){
                ans.set(j, ans.get(j - 1) + ans.get(j));
            }
        }
        return ans;
    }
    public static int maxProfit(int[] prices){
        int n = prices.length;
        int profit = 0, maxSell = prices[n-1], maxProfit = 0;
        for(int i = n - 2; i >=0; i--){
            if(prices[i] > maxSell)
                maxSell = prices[i];
            if(maxSell - prices[i] > 0)
                profit = maxSell - prices[i];
            maxProfit = Math.max(profit, maxProfit);
        }
        return maxProfit;
    }

    public boolean isPalindrome(String s){
        if(s.isEmpty())
            return true;
        int i = 0, j = s.length() - 1;
        char cI, cJ;
        while(i <= j){
            cI = s.charAt(i);
            cJ = s.charAt(j);
            if(!Character.isLetterOrDigit(cI))
                i++;
            else if(!Character.isLetterOrDigit(cJ))
                j--;
            else {
                if(Character.toLowerCase(cI) != Character.toLowerCase(cJ))
                    return false;

                i++;
                j--;
            }
        }
        return true;
    }

    public static int singleNumber(int[] nums){
        int a = 0;
        for(int i: nums)
            a = a ^ i;

        return a;
    }

    public boolean hasCycle(ListNode head){
        if(head == null)
            return false;
        ListNode slow_ptr = head;
        ListNode fast_ptr = head.next;

        while(slow_ptr != fast_ptr){
            if(fast_ptr == null || fast_ptr.next == null)
                return false;
            slow_ptr = slow_ptr.next;
            fast_ptr = fast_ptr.next.next;
        }
        return true;
    }

    List<Integer> list1 = new ArrayList<>();
    public List<Integer> preOrderTraversal(TreeNode root){
        if(root == null)
            return list1;
        else{
            list1.add(root.val);
            preOrderTraversal(root.left);
            preOrderTraversal(root.right);
        }
        return list1;
    }

    List<Integer> list2 = new ArrayList<>();
    public List<Integer> postOrderTraversal(TreeNode root){
        if(root == null)
            return list2;
        else{

            postOrderTraversal(root.left);
            postOrderTraversal(root.right);
            list2.add(root.val);
        }
        return list2;
    }


    class MinStack{
        Stack<Integer> s, temp;
        public MinStack(){
            s = new Stack<>();
            temp = new Stack<>();
        }
        public void push(int val){
            s.push(val);
            if(val <= getMin())
                temp.push(val);
        }
        public void pop(){
            int ele = s.pop();
            if(ele == getMin())
                temp.pop();
        }
        public int top(){
            return s.peek();
        }
        public int getMin(){
            if(temp.isEmpty())
                return Integer.MAX_VALUE;
            return temp.peek();
        }
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB){
        if(headA == null || headB == null)
            return null;

        ListNode a = headA;
        ListNode b = headB;

        while(a != b){
            a = a == null ? headB: a.next;
            b = b == null ? headA: b.next;
        }
        return a;
    }


    public static int[] twoSumII(int[] numbers, int target){
        int[] res = new int[2];

        int first = 0, second = numbers.length - 1;
        while(numbers[first]+ numbers[second] != target){
            if(numbers[first] + numbers[second] < target)
                first++;
            else
                second--;
        }
        res[0] = first;
        res[1] = second;
        return res;
    }

    public static String convertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();

        while(columnNumber > 0){
            columnNumber--;
            sb.insert(0,(char) ('A' + columnNumber % 26));
            columnNumber /= 26;
        }
        return  sb.toString();
    }

    public int majorityElement(int[] nums){
        int major = nums[0], count = 1;
        for(int i = 1; i < nums.length; i++){
            if(major == nums[i]){
                count++;
            } else if(count == 0){
                count++;
                major = nums[i];
            } else
                count--;
        }
        return major;
    }

    public static int titleToNumber(String columnTitle){
        int ans = 0;
        for(char c: columnTitle.toCharArray()){
            int i = c - 'A' + 1;
            ans = ans * 26 + i;
        }
        return ans;

    }
    public static int trailingZeroes(int n){
        if(n == 0)
            return 0;
        int ans = n/5;
        return ans + trailingZeroes(n/5);
    }

    public int hammingWeight(int n){
        int ans = 0;
        while(n != 0){
            ans = ans + (n & 1);
            n = n >>> 1;
        }
        return ans;
    }
    public boolean isHappy(int n){
        int rem, sum ;
        while(n > 9){
            sum = 0;
            while(n != 0){
                rem = n % 10;
                sum += Math.pow(rem, 2);
                n /= 10;
            }
            n = sum;
        }
        return n == 1 || n == 7;
    }

    public ListNode removeElements(ListNode head, int val){
        ListNode fakeHead = new ListNode(-1, head);
        ListNode prev = fakeHead;
        ListNode curr = head;
        while(curr != null){
            if(curr.val == val)
                prev.next = curr.next;
            else
                prev = curr;
            curr = curr.next;
        }
        return fakeHead.next;
    }

    public static int countPrimes(int n){
        if(n <= 1)
            return 0;
        boolean[] notPrime = new boolean[n];
        int count = 0;

        notPrime[0] = true;
        notPrime[1] = true;

        for(int i = 2; i < Math.sqrt(n); i++){
            if(!notPrime[i]){
                for(int j = 2; j*i < n; j++){
                    notPrime[i*j] = true;
                }
            }
        }
        for(int i = 2; i < notPrime.length; i++)
            if(!notPrime[i])
                count++;

        return count;
    }

    public boolean isIsomorphic(String s, String t){

        Map<Character, Character> map = new HashMap<>();
        for(int i = 0; i < s.length(); i++){
            char sI = s.charAt(i);
            char tI = t.charAt(i);

            if(map.containsKey(sI) && !map.get(sI).equals(tI))
                return false;
            else if(!map.containsKey(sI) && map.containsValue(tI))
                return false;

            map.put(sI, tI);
        }
        return true;
    }



    public ListNode reverseList(ListNode head){
        //recursive method
//        if(head == null || head.next == null)
//            return head;
//        ListNode tmp = reverseList(head.next);
//        ListNode prev = head.next;
//        prev.next = head;
//        head.next = null;
//
//        return tmp;

        //Iterative method
        ListNode prev = null;
        ListNode curr = head;
        while(curr != null){
            ListNode nextNode = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextNode;
        }
        return prev;
    }
    public boolean containsDuplicate(int[] nums){
        Set<Integer> set = new HashSet<>();
        for(int i: nums){
            if(!set.add(i))
                return true;
        }
        return false;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k){
        Map<Integer,Integer> count = new HashMap<>();

        for(int i = 0; i < nums.length; i++){

            if(count.containsKey(nums[i]) && (Math.abs(i - count.get(nums[i])) <= k)){
                return true;
            }
            count.put(nums[i], i);
        }
        return false;
    }

    public TreeNode invertTree(TreeNode root){

        if(root == null)
            return null;
        TreeNode temp = root.right;

        root.right = invertTree(root.left);
        root.left = invertTree(temp);

        return root;
    }

    public static List<String> summaryRanges(int[] nums){
        if(nums == null)
            return new ArrayList<>();

        List<String> list = new ArrayList<>();
        if(nums.length == 1){
            list.add(nums[0]+"");
            return list;
        }

        for(int i = 0; i < nums.length;){
            int a = nums[i];
            while(i+1 < nums.length && (nums[i+1] - nums[i]) == 1){
                i++;
            }
            if(a != nums[i])
                list.add(a+"->"+ nums[i]);
            else
                list.add(a+"");

            i++;
        }
        return list;
    }

    public boolean isPowerOfTwo(int n){

        return n > 0 && (n == 1 || (n % 2 == 0 && isPowerOfTwo(n/2)));
    }

    public ListNode reverseList1(ListNode node){
        ListNode prev = null;
        while(node != null){
            ListNode next = node.next;
            node.next = prev;
            prev = node;
            node = next;
        }
        return prev;
    }
    public boolean isPalindrome(ListNode head){
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        if(fast != null)
            slow = slow.next;
        slow = reverseList1(slow);
        fast = head;

        while(slow != null){
            if(slow.val != fast.val)
                return false;
            slow = slow.next;
            fast = fast.next;
        }
        return true;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        int parentVal = root.val;
        int pVal = p.val;
        int qVal = q.val;

        if(pVal > parentVal && qVal > parentVal)
            return lowestCommonAncestor(root.right, p, q);
        else if(pVal < parentVal && qVal < parentVal)
            return lowestCommonAncestor(root.left, p, q);
        else
            return root;
    }

    public void deleteNode(ListNode node){
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public boolean isAnagram(String s, String t){
        int[] alpha = new int[26];
        for(char c: s.toCharArray())
            alpha[c-'a']++;

        for(char c: t.toCharArray())
            alpha[c - 'a']--;

        for(int i: alpha){
            if(i != 0)
                return false;
        }
        return true;
    }

    public List<String> binaryTreePaths(TreeNode root){
        List<String> ans = new ArrayList<>();
        if(root == null)
            return ans;
        StringBuilder sb = new StringBuilder();
        dfs(root, ans, sb);
        return ans;
    }
    public static void dfs(TreeNode root, List<String> ans, StringBuilder sb){
        if(root.left == null && root.right == null){
            sb.append("" + root.val);
            ans.add(sb.toString());
            return;
        }

        if(root.left != null){
            String prev = sb.toString();
            sb.append(root.val + "->");
            dfs(root.left, ans, sb);
            sb = new StringBuilder(prev);
        }

        if(root.right != null){
            sb.append(root.val+ "->");
            dfs(root.right, ans, sb);
        }
    }

    public int addDigits(int num){
        //Iterative method
        // if(num < 10)
        //     return num;
        // while(num > 9){
        //     int sum = 0;
        //     while(num != 0){
        //         int rem = num % 10;
        //         sum += rem;
        //         num = num/10;
        //     }
        //     num = sum;
        // }
        // return num;

        //O(1)
        if(num == 0)
            return 0;
        else {
            if(num % 9 == 0)
                return 9;
            else
                return num % 9;
        }

    }

    public static int missingNumber(int[] nums){
//        int ans = 0;
//        for(int i = 0; i < nums.length; i++){
//
//            ans = ans ^ i ^ nums[i];
//        }
//        return ans ^ nums.length;

        int n = nums.length;
        int sum = (0 + n) * ( n + 1)/2;
        for(int i = 0; i < n; i++)
            sum -= nums[i];

        return sum;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2){
        if(root1 == null)
            return root2;
        if(root2 == null)
            return root1;

        root1.val += root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);

        return root1;
    }

    public String toLowerCase(String s){
       char[] chars = s.toCharArray();
        for(int i = 0; i < chars.length; i++){
            if(chars[i] >= 'A' && chars[i] <= 'Z')
                chars[i] += 32;
        }
        return String.valueOf(chars);
    }

    public boolean backspaceCompare(String s, String t){
        int i = s.length() - 1, j = t.length() - 1;
        int s_skips = 0;
        int t_skips = 0;

        while(i >= 0 || j >= 0){

            while(i >= 0){
                if(s.charAt(i) == '#') {
                    s_skips++;
                    i--;
                } else if(s_skips > 0){
                    i--;
                    s_skips--;
                } else
                    break;
            }

            while(j >= 0){
                if(s.charAt(j) == '#') {
                    t_skips++;
                    j--;
                } else if(t_skips > 0){
                    j--;
                    t_skips--;
                } else
                    break;
            }

            if(i >= 0 && j >= 0 && s.charAt(i) != t.charAt(j) )
                return false;

            if((i >= 0) != (j >= 0))
                return false;
            i--;
            j--;
        }
        return true;
    }

    public void reverseString(char[] s){
        int i = 0;
        int j = s.length - 1;

        while(i <= j){
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;

            i++;
            j--;
        }
    }

    public ListNode middleNode(ListNode head){
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public boolean judgeCircle(String moves){

        int x = 0;
        int y = 0;

        for(char ch: moves.toCharArray()){
            if(ch == 'U')
                y++;
            else if(ch == 'D')
                y--;
            else if(ch == 'L')
                x--;
            else
                x++;
        }
        return x == 0 && y == 0;
    }

    public int[] sortedSquares(int[] nums){
        int n = nums.length;
        int pos = 0;

        while(pos < n && nums[pos] < 0)
            pos++;
        int neg = pos - 1;

        int[] ans = new int[n];
        int counter = 0;
        while(neg >= 0 && pos < n){
            if(nums[neg] * nums[neg] < nums[pos] * nums[pos]){
                ans[counter++] = nums[neg] * nums[neg];
                neg--;
            } else{
                ans[counter++] = nums[pos] * nums[pos];
                pos++;
            }
        }
        while(neg >= 0){
            ans[counter++] = nums[neg] * nums[neg];
            neg--;
        }
        while(pos < n){
            ans[counter++] = nums[pos] * nums[pos];
            pos++;
        }
        return ans;
    }

    public int rangeSumBST(Medium.TreeNode root, int low, int high){
        if(root == null)
            return 0;

        if(root.val < low)
            return rangeSumBST(root.right, low, high);
        if(root.val > high)
            return rangeSumBST(root.left, low, high);

        return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high);
    }

    public boolean isUnivalTree(Medium.TreeNode root){

        boolean left = root.left == null || root.left.val == root.val && isUnivalTree(root.left);
        boolean right = root.right == null || root.val == root.right.val && isUnivalTree(root.right);

        return left && right;
    }

    public int peakIndexInMountainArray(int[] arr){
        int left = 0, right = arr.length - 1;
        while(left <= right){
            int mid = left + (right - left)/2;
            if(arr[mid] < arr[mid + 1])
                left = mid + 1;
            else
                right = mid - 1;
        }
        return left;
    }

    public int sumOfLeftLeaves(TreeNode root){

        if(root == null)
            return 0;
        int ans = 0;
        if(root.left != null){
            if(root.left.left == null && root.left.right == null)
                ans += root.left.val;
            else
                ans += sumOfLeftLeaves(root.left);
        }
        ans += sumOfLeftLeaves(root.right);

        return ans;
    }

    public int arrayPairSum(int[] nums){
        Arrays.sort(nums);
        int sum = 0;
        for(int i = 0; i < nums.length; i+= 2)
            sum += nums[i];
        return sum;
    }

    public List<String> commonChars(String[] words){

        List<String> ans = new ArrayList<>();

        int[] min_freq = new int[26];
        Arrays.fill(min_freq, Integer.MAX_VALUE);

        for(String s: words){

            int[] char_freq = new int[26];

            for(char c: s.toCharArray()){
                char_freq[c - 'a']++;
            }

            for(int i = 0; i < 26; i++){
                min_freq[i] = Math.min(min_freq[i], char_freq[i]);
            }
        }

        for(int i = 0; i < 26; i++){
            while(min_freq[i] > 0){
                ans.add("" + (char)(i + 'a'));
                min_freq[i]--;
            }
        }
        return ans;
    }

    public int[] sortArrayByParityII(int[] nums){
        int i = 0, j = 1;
        int n = nums.length;

        while(i < n && j < n){

            while(i < n && nums[i] % 2 == 0)
                i+= 2;

            while(j < n && nums[j] % 2 != 0)
                j+= 2;

            if(i < n && j < n){
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }
        return nums;
    }

    public int maxProfitII(int[] prices){

        int n = prices.length - 1, buy = 0, sell = 0, profit = 0, i = 0;
        while(i < n){
            while(i < n && prices[i] >= prices[i + 1])
                i++;
            buy = prices[i];

            while(i < n && prices[i] < prices[i+1])
                i++;
            sell = prices[i];

            profit += (sell - buy);
        }
        return profit;
    }

    public static String removeDuplicates(String s){
        int k = 0, n = s.length();
        char[] res = s.toCharArray();
        for(int i = 0; i < n ; i++){
           res[k] = res[i];
           if(k > 0 && res[k - 1] == res[k])
               k -=2;
           k++;
        }

        return new String(res, 0, k);
    }

    public int islandPerimeter(int[][] grid){
        int result = 0;

        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    result += 4;

                    if(i > 0 && grid[i - 1][j] == 1)
                        result -= 2;

                    if(j > 0 && grid[i][j - 1] == 1)
                        result -= 2;
                }
            }
        }
        return result;
    }

    public boolean rotateString(String s, String goal){
        return s.length() == goal.length() && (s + s).contains(goal);
    }
    public static void main(String[] args) {

//        int[] nums = {2,7,11,15};
//        int target = 9;
//        int[] ans = twoSum(nums, target);
//        for(int i: ans){
//            System.out.print(i + " ");
//        }

//        int num = 123;
//        System.out.println(reverse(num));
        //System.out.println(Integer.MAX_VALUE + " " + Integer.MIN_VALUE);
//        String[] sample = {"flower", "flow", "flight"};
//        System.out.println(longestCommonPrefix(sample));
       // System.out.println("flow".indexOf("flower"));
        //System.out.println(isValid("([}}])"));
        //System.out.println(addBinary("11", "1"));
//        int arr[] = {2,1,4};
//        System.out.println(maxProfit(arr));
        //System.out.println(singleNumber(new int[]{4,1,2,1,2}));
        //System.out.println(convertToTitle(28));
        //System.out.println(titleToNumber("AB"));
        //System.out.println(trailingZeroes(30));
        //System.out.println(countPrimes(10));
        //System.out.println(summaryRanges(new int[]{0,1,2,4,5,7}));
        //System.out.println(missingNumber(new int[]{0,1}));
        //System.out.println(new Easy().generate(5));
//        String[] words = {"bella", "label", "roller"};
//        System.out.println(new Easy().commonChars(words));
        System.out.println(removeDuplicates("abbaca"));
    }
}
