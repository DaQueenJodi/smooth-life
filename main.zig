const std = @import("std");
const c = @import("c.zig");
const Allocator = std.mem.Allocator;

const clamp = std.math.clamp;

const NPROC = 2;

const USING_SDL = false;
const USING_RAYLIB = !USING_SDL;

const WINDOW_WIDTH = CELL_SIZE*GRID_WIDTH;
const WINDOW_HEIGHT = CELL_SIZE*GRID_HEIGHT;
const CELL_SIZE = 4;
const GRID_WIDTH = 200;
const GRID_HEIGHT = 200;

const radius_ratio = 3.0;
const outer_radius = 21.0;
const inner_radius = outer_radius/radius_ratio;
// M and N are just areas of the circle/ring
const M = std.math.pi * inner_radius * inner_radius;
const N = std.math.pi * outer_radius * outer_radius;
const b_1 = 0.278;
const b_2 = 0.365;
const d_1 = 0.267;
const d_2 = 0.445;
const alpha_n = 0.028;
const alpha_m = 0.147;
const delta_time = 0.05;

var grid: [GRID_HEIGHT*GRID_WIDTH]f32 = undefined;
var grid_copy: [GRID_HEIGHT*GRID_WIDTH]f32 = undefined;

fn sigmoid_1(x: f32, a: f32, alpha: f32) f32 {
    return 1.0/(1.0 + std.math.exp(-(x - a)*4/alpha));
}
fn sigmoid_2(x: f32, a: f32, b: f32) f32 {
    return sigmoid_1(x, a, alpha_n)*(1 - sigmoid_1(x, b, alpha_n));
}
fn sigmoid_m(x: f32, y: f32, m: f32) f32 {
    return x*(1 - sigmoid_1(m, 0.5, alpha_m)) + y*sigmoid_1(m, 0.5, alpha_m);
}
fn s(n: f32, m: f32) f32 {
    return sigmoid_2(n, sigmoid_m(b_1, d_1, m), sigmoid_m(b_2, d_2, m));
}

fn calculate_new_value(x: usize, y: usize) f32 {
    const x_i: i32 = @intCast(x);
    const y_i: i32 = @intCast(y);

    var n: f32 = 0;
    var m: f32 = 0;

    // zig doesn't support ranges that aren't usize :(
    var dy: i32 = -(outer_radius - 1);
    while (dy < outer_radius) : (dy += 1) {
        var dx: i32 = -outer_radius;
        while (dx < outer_radius) : (dx += 1) {
            const new_y: usize = @intCast(@mod(y_i+dy, GRID_HEIGHT));
            const new_x: usize = @intCast(@mod( x_i+dx, GRID_WIDTH));
            const t: f32 = @floatFromInt(dx*dx + dy*dy);
            if (t <= inner_radius*inner_radius) {
                m += grid[new_y*GRID_HEIGHT+new_x];
            } else if (t <= outer_radius*outer_radius) {
                n += grid[new_y*GRID_HEIGHT+new_x];
            }
        }
    }
    const res = s(n/N, m/M);
    return res*2.0 - 1;
}

fn workerFunction(x: usize, y: usize) void {
    const cell = &grid[y*GRID_HEIGHT+x];
    const new_val = calculate_new_value(x, y);
    cell.* = clamp(cell.* + delta_time*new_val, 0.0, 1.0);
}

fn next(allocator: Allocator) !void {
    grid_copy = grid;

    var threads: [NPROC]std.Thread = undefined;
    var pool = std.Thread.Pool{.allocator = allocator, .threads = &threads};
    try pool.init(.{.allocator = allocator, .n_jobs = NPROC});
    defer pool.deinit();
    inline for (0..GRID_HEIGHT) |y| {
        for (0..GRID_WIDTH) |x| {
            try pool.spawn(workerFunction, .{x, y});
        }
    }
    var wait_group = std.Thread.WaitGroup{};
    pool.waitAndWork(&wait_group);
}

fn init_grid() !void {
    for (&grid) |*cell| cell.* = 0.1;
    var seed: u64 = undefined;
    try std.os.getrandom(std.mem.asBytes(&seed));
    var pcg = std.rand.Pcg.init(seed);
    const rand = pcg.random();
    //for (grid) |*cell| cell.* = rand.float(f32);
    for (GRID_HEIGHT/2..GRID_HEIGHT) |y| {
        for (GRID_WIDTH/2..GRID_WIDTH*3/4) |x| {
            grid[y*GRID_HEIGHT+x] = clamp(rand.float(f32)*1.5, 0.0, 1.0);
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    try init_grid();
    c.SetTraceLogLevel(c.LOG_FATAL | c.LOG_WARNING | c.LOG_ERROR);
    c.SetTargetFPS(60);
    c.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "welp");
    for (0..100) |_| {
        c.BeginDrawing();
        for (0..GRID_HEIGHT) |y| {
            for (0..GRID_WIDTH) |x| {
                const cell = grid[y*GRID_HEIGHT+x];
                const whiteness: u8 = @intFromFloat(@floor(255 * cell));
                if (CELL_SIZE == 1) {
                    c.DrawPixel(@intCast(x*CELL_SIZE), @intCast(y*CELL_SIZE), c.Color{.r = whiteness, .g = whiteness, .b = whiteness, .a = 255});
                } else {
                    c.DrawRectangle(@intCast(x*CELL_SIZE), @intCast(y*CELL_SIZE), CELL_SIZE, CELL_SIZE, c.Color{.r = whiteness, .g = whiteness, .b = whiteness, .a = 255});
                }
            }
        }
        c.EndDrawing();
        //if (c.IsKeyPressed(c.KEY_R)) try init_grid();
        try next(allocator);
    }
    c.CloseWindow();
}
