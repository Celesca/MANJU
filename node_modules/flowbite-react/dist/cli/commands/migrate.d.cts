export interface TransformResult {
    content: string;
    changed: boolean;
}
export interface Transformer {
    name: string;
    transform: (content: string) => TransformResult;
}
export declare function migrate(): Promise<void>;
